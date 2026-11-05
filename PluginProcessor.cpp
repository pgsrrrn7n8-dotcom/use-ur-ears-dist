#include "PluginProcessor.h"

namespace
{
    constexpr double HALFBAND_STAGE1[8] = {
        0.04063346092419326, 0.1505051290226746, 0.3007570559918741, 0.4607745049614506,
        0.6095243148961883, 0.7385038411188573, 0.8492238103920661, 0.9497427837050002
    };
    constexpr double HALFBAND_STAGE2[4] = {
        0.06069013102679145, 0.22801137585234285, 0.474337659974904, 0.7955654775554291
    };
    constexpr double SHAPER_TRIM[4] = { 0.95, 0.92, 1.0, 0.9 };
    constexpr double SMOOTH_COEF = 0.002;
    constexpr double MAX_PAN = 0.85;
    const double LIMIT_CEILING = std::pow(10.0, -0.1 / 20.0);
    const double Q_NORM = 1.0 / std::tanh(1.5);

    std::vector<double> designHalfPiCoefs(int count, double transition)
    {
        double k = std::tan((1.0 - transition * 2.0) * juce::MathConstants<double>::pi / 4.0);
        k *= k;
        double kk = std::pow(1.0 - k * k, 0.25);
        double e = 0.5 * (1.0 - kk) / (1.0 + kk);
        double e2 = e * e, e4 = e2 * e2;
        double q = e * (1.0 + e4 * (2.0 + e4 * (15.0 + 150.0 * e4)));
        int order = count * 2 + 1;
        std::vector<double> out;
        for (int idx = 0; idx < count; ++idx) {
            int c = idx + 1;
            double num = 0.0, den = 0.0, term = 0.0;
            int i = 0, sgn = 1;
            do {
                term = std::pow(q, i * (i + 1)) * std::sin((i * 2 + 1) * c * juce::MathConstants<double>::pi / order) * sgn;
                num += term; sgn = -sgn; ++i;
            } while (std::abs(term) > 1e-100);
            i = 1; sgn = -1;
            do {
                term = std::pow(q, i * i) * std::cos(i * 2 * c * juce::MathConstants<double>::pi / order) * sgn;
                den += term; sgn = -sgn; ++i;
            } while (std::abs(term) > 1e-100);
            double ww = (num * std::pow(q, 0.25)) / (den + 0.5);
            double w2 = ww * ww;
            double x = std::sqrt((1.0 - w2 * k) * (1.0 - w2 / k)) / (1.0 + w2);
            out.push_back((1.0 - x) / (1.0 + x));
        }
        return out;
    }

    inline double hermite(const float* buf, int mask, double pos)
    {
        int i0 = (int)std::floor(pos);
        double f = pos - i0;
        double xm1 = buf[(i0 - 1) & mask];
        double x0  = buf[i0 & mask];
        double x1  = buf[(i0 + 1) & mask];
        double x2  = buf[(i0 + 2) & mask];
        double c1 = 0.5 * (x1 - xm1);
        double c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2;
        double c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1);
        return ((c3 * f + c2) * f + c1) * f + x0;
    }

    inline double shapeBrutal(double x)   { return x > 1.0 ? 1.0 : (x < -0.72 ? -0.72 : x); }
    inline double shapePlasma(double x)   { return 0.72 * std::tanh(1.6 * std::sin(1.5707963267948966 * x)) + 0.28 * std::tanh(x); }
    inline double shapeDiode(double x)    { return x >= 0.0 ? 1.0 - std::exp(-1.5 * x) : -0.6 * (1.0 - std::exp(3.2 * x)); }
    inline double shapeQuantize(double x) {
        double t = std::tanh(1.2 * x);
        double q = std::round(t * 7.0) / 7.0;
        return std::tanh(1.5 * q) * Q_NORM;
    }
}

void ClashDollProcessor::HilbertIQ::init(double sampleRate)
{
    std::fill(std::begin(mi), std::end(mi), 0.0);
    std::fill(std::begin(mq), std::end(mq), 0.0);
    prev = i = q = 0.0;
    int sr = (int)std::round(sampleRate);
    if (sr == 44100) {
        double ti[4] = { 0.17330783718372933, 0.7554498953106368, 0.9536447683431815, 0.9925158705350553 };
        double tq[4] = { 0.5035658412897742, 0.8910202296720034, 0.9808192771139574, 0.9980447985729857 };
        std::copy(ti, ti + 4, ci); std::copy(tq, tq + 4, cq);
    } else if (sr == 48000) {
        double ti[4] = { 0.17674852306419814, 0.7617367254502089, 0.9558425547957876, 0.9930031487316011 };
        double tq[4] = { 0.5105640214317815, 0.8950477393554356, 0.9819123562931757, 0.9981819840118007 };
        std::copy(ti, ti + 4, ci); std::copy(tq, tq + 4, cq);
    } else {
        auto c = designHalfPiCoefs(8, 30.0 / sampleRate);
        for (int k = 0; k < 4; ++k) { ci[k] = c[k * 2]; cq[k] = c[k * 2 + 1]; }
    }
}

double ClashDollProcessor::HilbertIQ::runStage(double x, const double* c, double* m)
{
    double a = x;
    for (int s = 0; s < 4; ++s) {
        int o = s * 4;
        double y = c[s] * (a + m[o + 3]) - m[o + 1];
        m[o + 1] = m[o]; m[o] = a;
        m[o + 3] = m[o + 2]; m[o + 2] = y;
        a = y;
    }
    return a;
}

void ClashDollProcessor::HilbertIQ::process(double x)
{
    i = runStage(x, ci, mi);
    double delayed = prev;
    prev = x;
    q = runStage(delayed, cq, mq);
}

juce::AudioProcessorValueTreeState::ParameterLayout ClashDollProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;

    // Cubic-like skew around 0 for Hz Shift
    juce::NormalisableRange<float> hzRange(-40.0f, 40.0f,
        [](float, float, float norm) {
            float u = 2.0f * norm - 1.0f;
            return (u >= 0.0f ? 1.0f : -1.0f) * std::pow(std::abs(u), 3.0f) * 40.0f;
        },
        [](float, float, float val) {
            float u = juce::jlimit(-1.0f, 1.0f, val / 40.0f);
            return ((u >= 0.0f ? 1.0f : -1.0f) * std::cbrt(std::abs(u)) + 1.0f) * 0.5f;
        });

    p.push_back(std::make_unique<juce::AudioParameterFloat>("hz", "HZ SHIFT", hzRange, 3.2f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("cents", "CENTS", juce::NormalisableRange<float>(-100.0f, 100.0f, 1.0f), 35.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("grain", "GRAIN", juce::NormalisableRange<float>(6.0f, 12.0f, 0.1f), 8.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("grind", "FM GRIND", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.3f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("dupe", "DUPE MIX", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.8f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("split", "STEREO SPLIT", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.15f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("focus", "CLASH FOCUS", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.4f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("drive", "DRIVE", juce::NormalisableRange<float>(0.0f, 36.0f, 0.1f), 20.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("bias", "BIAS SPREAD", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.35f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("sub", "SUB SAFE 120", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.6f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("tone", "TONE TILT", juce::NormalisableRange<float>(-1.0f, 1.0f, 0.001f), 0.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("mix", "DRY / WET", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f));
    p.push_back(std::make_unique<juce::AudioParameterFloat>("out", "OUTPUT", juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f), -6.0f));
    p.push_back(std::make_unique<juce::AudioParameterChoice>("pitchMode", "Pitch Mode", juce::StringArray{ "HZ SHIFT", "CENTS", "FM GRIND" }, 1));
    p.push_back(std::make_unique<juce::AudioParameterChoice>("distMode", "Dist Mode", juce::StringArray{ "BRUTAL CLIP", "PLASMA FOLD", "DIODE CRUNCH", "QUANTIZE RIPPER" }, 0));

    return { p.begin(), p.end() };
}

ClashDollProcessor::ClashDollProcessor()
    : AudioProcessor(BusesProperties().withInput("Input", juce::AudioChannelSet::stereo(), true)
                                      .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout())
{
}

bool ClashDollProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return layouts.getMainInputChannelSet() == out;
}

void ClashDollProcessor::prepareToPlay (double sampleRate, int)
{
    fs = sampleRate > 0.0 ? sampleRate : 48000.0;
    hil[0].init(fs); hil[1].init(fs);
    theta = 0.0;

    dlSize = fs > 96000.0 ? 32768 : 16384;
    dlMask = dlSize - 1;
    dl[0].assign(dlSize, 0.0f);
    dl[1].assign(dlSize, 0.0f);
    dlW = 0; psPhase = 0.0;

    for (int c = 0; c < 2; ++c) {
        pre[c].reset(); post[c].reset();
        up1[c].init(HALFBAND_STAGE1, 8);
        up2[c].init(HALFBAND_STAGE2, 4);
        dn2[c].init(HALFBAND_STAGE2, 4);
        dn1[c].init(HALFBAND_STAGE1, 8);
        dcX[c] = dcY[c] = 0.0;
        for (int k = 0; k < 2; ++k) {
            wetLp[c][k].reset(); wetLp[c][k].setLowpass(fs, 120.0, juce::MathConstants<double>::sqrt2 * 0.5);
            wetHp[c][k].reset(); wetHp[c][k].setHighpass(fs, 120.0, juce::MathConstants<double>::sqrt2 * 0.5);
        }
        cleanLp[c].reset(); cleanLp[c].setLowpass(fs, 120.0, juce::MathConstants<double>::sqrt2 * 0.5);
        tiltLp[c] = 0.0;
    }
    lastFocusDb = -100.0;
    dcR4 = std::exp(-TWO_PI * 15.0 / (fs * 4.0));
    dcR1 = std::exp(-TWO_PI * 15.0 / fs);
    cleanDcX = cleanDcY = 0.0;
    tiltA = 1.0 - std::exp(-TWO_PI * 650.0 / fs);
    limEnv = 1.0;
    limRel = 1.0 - std::exp(-1.0 / (0.06 * fs));

    const double initVals[SMOOTH_COUNT] = {
        0.0, 46.0, 8.0, 0.0, 0.85, 0.18, 0.55, std::pow(10.0, 22.0 / 20.0), 0.35,
        0.6, 0.15, 1.0, std::pow(10.0, -6.0 / 20.0), 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0
    };
    for (int i = 0; i < SMOOTH_COUNT; ++i) s[i] = t[i] = initVals[i];
}

void ClashDollProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numCh = buffer.getNumChannels();
    const int n = buffer.getNumSamples();
    if (numCh == 0 || n == 0) return;

    t[S_HZ]    = apvts.getRawParameterValue("hz")->load();
    t[S_CENTS] = apvts.getRawParameterValue("cents")->load();
    t[S_GRAIN] = apvts.getRawParameterValue("grain")->load();
    t[S_GRIND] = apvts.getRawParameterValue("grind")->load();
    t[S_DUPE]  = apvts.getRawParameterValue("dupe")->load();
    t[S_SPLIT] = apvts.getRawParameterValue("split")->load();
    t[S_FOCUS] = apvts.getRawParameterValue("focus")->load();
    t[S_DRIVE] = std::pow(10.0, (double)apvts.getRawParameterValue("drive")->load() / 20.0);
    t[S_BIAS]  = apvts.getRawParameterValue("bias")->load();
    t[S_SUB]   = apvts.getRawParameterValue("sub")->load();
    t[S_TONE]  = apvts.getRawParameterValue("tone")->load();
    t[S_MIX]   = apvts.getRawParameterValue("mix")->load();
    t[S_OUT]   = std::pow(10.0, (double)apvts.getRawParameterValue("out")->load() / 20.0);

    int pm = (int)std::round(apvts.getRawParameterValue("pitchMode")->load());
    int dm = (int)std::round(apvts.getRawParameterValue("distMode")->load());
    for (int i = 0; i < 3; ++i) t[S_PM0 + i] = (i == pm) ? 1.0 : 0.0;
    for (int i = 0; i < 4; ++i) t[S_DM0 + i] = (i == dm) ? 1.0 : 0.0;

    float* chL = buffer.getWritePointer(0);
    float* chR = numCh > 1 ? buffer.getWritePointer(1) : nullptr;

    double b2[2]{}, b4[4]{};
    const double thetaInc = TWO_PI / fs;
    float pkL = 0.0f, pkR = 0.0f, minGr = 1.0f;
    int vPos = visWritePos.load(std::memory_order_relaxed);

    for (int i = 0; i < n; ++i)
    {
        for (int p = 0; p < SMOOTH_COUNT; ++p)
            s[p] += (t[p] - s[p]) * SMOOTH_COEF;

        if ((i & 15) == 0) {
            double focusDb = s[S_FOCUS] * 18.0;
            if (std::abs(focusDb - lastFocusDb) > 0.005) {
                lastFocusDb = focusDb;
                for (int c = 0; c < 2; ++c) {
                    pre[c].setPeak(fs, 1450.0, 1.0, focusDb);
                    post[c].setPeak(fs, 1450.0, 1.0, -focusDb);
                }
            }
            double tiltDb = s[S_TONE] * 9.0;
            tiltHigh = std::pow(10.0, tiltDb / 20.0);
            tiltLow = 1.0 / tiltHigh;
        }

        double xL = chL[i];
        double xR = chR ? chR[i] : xL;

        hil[0].process(xL + ANTI_DENORMAL);
        hil[1].process(xR + ANTI_DENORMAL);
        double iL = hil[0].i, qL = hil[0].q;
        double iR = hil[1].i, qR = hil[1].q;

        theta += thetaInc * s[S_HZ];
        if (theta > juce::MathConstants<double>::pi) theta -= TWO_PI;
        else if (theta < -juce::MathConstants<double>::pi) theta += TWO_PI;

        int w = dlW;
        dl[0][w] = (float)iL;
        dl[1][w] = (float)iR;

        double v2L = 0.0, v2R = 0.0;
        double wm0 = s[S_PM0], wm1 = s[S_PM1], wm2 = s[S_PM2];

        if (wm0 > 1e-4) {
            double ct = std::cos(theta), st = std::sin(theta);
            v2L += wm0 * (iL * ct - qL * st);
            v2R += wm0 * (iR * ct - qR * st);
        }

        {
            double winSamples = s[S_GRAIN] * 0.001 * fs;
            double ratio = std::pow(2.0, s[S_CENTS] / 1200.0);
            psPhase += (1.0 - ratio) / winSamples;
            psPhase -= std::floor(psPhase);
            if (wm1 > 1e-4) {
                double ph1 = psPhase;
                double ph2 = ph1 + 0.5 >= 1.0 ? ph1 - 0.5 : ph1 + 0.5;
                double g1 = 0.5 * (1.0 - std::cos(TWO_PI * ph1));
                double g2 = 1.0 - g1;
                double r1 = w - (ph1 * winSamples + 2.0);
                double r2 = w - (ph2 * winSamples + 2.0);
                v2L += wm1 * (g1 * hermite(dl[0].data(), dlMask, r1) + g2 * hermite(dl[0].data(), dlMask, r2));
                v2R += wm1 * (g1 * hermite(dl[1].data(), dlMask, r1) + g2 * hermite(dl[1].data(), dlMask, r2));
            }
        }

        if (wm2 > 1e-4) {
            double depth = s[S_GRIND] * 3.0 * juce::MathConstants<double>::pi;
            double phL = theta + depth * iL;
            double phR = theta + depth * iR;
            v2L += wm2 * (iL * std::cos(phL) - qL * std::sin(phL));
            v2R += wm2 * (iR * std::cos(phR) - qR * std::sin(phR));
        }

        dlW = (w + 1) & dlMask;

        double pan = s[S_SPLIT] * MAX_PAN;
        double ang = (pan + 1.0) * 0.7853981633974483;
        double gNear = juce::MathConstants<double>::sqrt2 * std::sin(ang);
        double gFar  = juce::MathConstants<double>::sqrt2 * std::cos(ang);
        double dupe = s[S_DUPE];
        double sumL = iL * gNear + dupe * v2L * gFar;
        double sumR = iR * gFar  + dupe * v2R * gNear;

        double drive = s[S_DRIVE];
        double bias  = s[S_BIAS];
        double wd0 = s[S_DM0], wd1 = s[S_DM1], wd2 = s[S_DM2], wd3 = s[S_DM3];

        double wetL = 0.0, wetR = 0.0;
        for (int c = 0; c < 2; ++c) {
            double x = (c == 0 ? sumL : sumR);
            double emph = pre[c].process(x);
            double driven = emph * drive + (c == 0 ? bias : -bias) + ANTI_DENORMAL;
            up1[c].process(driven, b2, 0);
            up2[c].process(b2[0], b4, 0);
            up2[c].process(b2[1], b4, 2);

            double dx = dcX[c], dy = dcY[c];
            for (int j = 0; j < 4; ++j) {
                double v = b4[j];
                double y = 0.0;
                if (wd0 > 1e-4) y += wd0 * SHAPER_TRIM[0] * shapeBrutal(v);
                if (wd1 > 1e-4) y += wd1 * SHAPER_TRIM[1] * shapePlasma(v);
                if (wd2 > 1e-4) y += wd2 * SHAPER_TRIM[2] * shapeDiode(v);
                if (wd3 > 1e-4) y += wd3 * SHAPER_TRIM[3] * shapeQuantize(v);
                double hp = y - dx + dcR4 * dy;
                dx = y; dy = hp;
                b4[j] = hp;
            }
            dcX[c] = dx; dcY[c] = dy;

            double d0 = dn2[c].process(b4[0], b4[1]);
            double d1 = dn2[c].process(b4[2], b4[3]);
            double ds = dn1[c].process(d0, d1);
            double deemph = post[c].process(ds);
            if (c == 0) wetL = deemph; else wetR = deemph;
        }

        double mono = 0.5 * (iL + iR);
        double cleanHp = mono - cleanDcX + dcR1 * cleanDcY;
        cleanDcX = mono; cleanDcY = cleanHp;
        double cleanLow = cleanLp[1].process(cleanLp[0].process(cleanHp));
        double sub = s[S_SUB];
        double subGain = std::min(drive, 2.0);

        double lowL  = wetLp[0][1].process(wetLp[0][0].process(wetL));
        double highL = wetHp[0][1].process(wetHp[0][0].process(wetL));
        double lowR  = wetLp[1][1].process(wetLp[1][0].process(wetR));
        double highR = wetHp[1][1].process(wetHp[1][0].process(wetR));
        double lowMono = 0.5 * (lowL + lowR);
        double safeLow = lowMono + (subGain * cleanLow - lowMono) * sub;
        double guardL = highL + lowL + (safeLow - lowL) * sub;
        double guardR = highR + lowR + (safeLow - lowR) * sub;

        tiltLp[0] += tiltA * (guardL - tiltLp[0]);
        tiltLp[1] += tiltA * (guardR - tiltLp[1]);
        double toneL = tiltLp[0] * tiltLow + (guardL - tiltLp[0]) * tiltHigh;
        double toneR = tiltLp[1] * tiltLow + (guardR - tiltLp[1]) * tiltHigh;

        double mix = s[S_MIX];
        double og  = s[S_OUT];
        double oL = (iL * (1.0 - mix) + toneL * mix) * og;
        double oR = (iR * (1.0 - mix) + toneR * mix) * og;

        double pk = std::max(std::abs(oL), std::abs(oR));
        double target = pk > LIMIT_CEILING ? LIMIT_CEILING / pk : 1.0;
        if (target < limEnv) limEnv = target;
        else limEnv += (target - limEnv) * limRel;

        oL = juce::jlimit(-LIMIT_CEILING, LIMIT_CEILING, oL * limEnv);
        oR = juce::jlimit(-LIMIT_CEILING, LIMIT_CEILING, oR * limEnv);

        chL[i] = (float)oL;
        if (chR) chR[i] = (float)oR;

        pkL = std::max(pkL, std::abs((float)oL));
        pkR = std::max(pkR, std::abs((float)oR));
        minGr = std::min(minGr, (float)limEnv);

        visL[vPos] = (float)oL;
        visR[vPos] = (float)oR;
        vPos = (vPos + 1) & (kScopeSize - 1);
    }

    visWritePos.store(vPos, std::memory_order_relaxed);
    meterPeakL.store(pkL, std::memory_order_relaxed);
    meterPeakR.store(pkR, std::memory_order_relaxed);
    meterGr.store(minGr, std::memory_order_relaxed);
}

void ClashDollProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void ClashDollProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
        if (xml->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

// ==============================================================================
// MATTE BOUTIQUE PINK UI (ZERO NEON GLOW + 14% HEAVIER KNOBS)
// ==============================================================================
namespace
{
    const juce::Colour colObsidian (0xff1a0914);
    const juce::Colour colPanel    (0xff24101d);
    const juce::Colour colPanel2   (0xff2e1426);
    const juce::Colour colEdge     (0xff4a1f3c);
    const juce::Colour colBubble   (0xffff85c0);
    const juce::Colour colMagenta  (0xffff007f);
    const juce::Colour colPastel   (0xffffd6eb);
    const juce::Colour colInk      (0xffffe9f4);
    const juce::Colour colMuted    (0xffb98aa6);

    struct PresetData {
        const char* name;
        int pitchMode; float hz, cents, grain, grind, dupe, split, focus, drive, bias;
        int distMode;  float sub, tone, mix, out;
    };

    constexpr PresetData PRESETS[5] = {
        { "Ne Plus Ultra Drop",    1,  0.0f,  46.0f,  8.0f, 0.0f,  0.85f, 0.18f, 0.55f, 22.0f, 0.35f, 0, 0.6f,  0.15f, 1.0f, -6.0f },
        { "Mono Intermod Growl",   0,  3.7f,   0.0f,  8.0f, 0.0f,  1.00f, 0.00f, 0.70f, 28.0f, 0.00f, 2, 0.4f, -0.10f, 1.0f, -6.0f },
        { "kHs Wide Bias Saw",     0,  0.35f,  0.0f,  8.0f, 0.0f,  0.50f, 0.10f, 0.20f, 18.0f, 0.80f, 1, 0.7f,  0.00f, 1.0f, -6.0f },
        { "Metallic Hz Ripper",    2, 17.5f,   0.0f,  8.0f, 0.45f, 0.90f, 0.35f, 0.80f, 26.0f, 0.25f, 3, 0.3f,  0.30f, 1.0f, -6.0f },
        { "Sub-Safe Riff Crusher", 1,  0.0f, -21.0f, 11.0f, 0.0f,  0.75f, 0.25f, 0.45f, 34.0f, 0.50f, 0, 1.0f,  0.05f, 1.0f, -4.0f }
    };

    class MatteKnob : public juce::Component
    {
    public:
        MatteKnob(juce::AudioProcessorValueTreeState& state, const juce::String& paramID,
                  const juce::String& labelText, bool bipolar, std::function<juce::String(float)> formatter)
            : apvts(state), id(paramID), label(labelText), isBi(bipolar), fmt(std::move(formatter))
        {
            param = apvts.getParameter(id);
            attachment = std::make_unique<juce::ParameterAttachment>(*param, [this](float) { repaint(); }, nullptr);
            attachment->sendInitialUpdate();
        }

        void setDimmed(bool d) { if (dimmed != d) { dimmed = d; setAlpha(dimmed ? 0.32f : 1.0f); repaint(); } }

        void paint(juce::Graphics& g) override
        {
            auto b = getLocalBounds().toFloat();
            const float cx = b.getCentreX();
            const float cy = 38.0f;
            const float norm = param ? param->getValue() : 0.0f;
            const float val  = param ? param->convertFrom0to1(norm) : 0.0f;

            const float startAng = -juce::MathConstants<float>::pi * 0.75f;
            const float endAng   =  juce::MathConstants<float>::pi * 0.75f;
            const float curAng   = startAng + norm * (endAng - startAng);
            const float zeroAng  = isBi ? 0.0f : startAng;

            // Ticks
            for (int i = 0; i <= 10; ++i) {
                float a = startAng + (i / 10.0f) * (endAng - startAng);
                float r0 = 34.0f, r1 = (i % 5 == 0) ? 38.0f : 36.5f;
                g.setColour(i % 5 == 0 ? colBubble : colEdge);
                g.drawLine(cx + r0 * std::sin(a), cy - r0 * std::cos(a),
                           cx + r1 * std::sin(a), cy - r1 * std::cos(a), 1.4f);
            }

            // Background Arc
            juce::Path bgArc;
            bgArc.addCentredArc(cx, cy, 30.0f, 30.0f, 0.0f, startAng, endAng, true);
            g.setColour(juce::Colour(0xff3a1430));
            g.strokePath(bgArc, juce::PathStrokeType(4.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

            // Active Solid Matte Arc (No Neon Blur)
            if (std::abs(curAng - zeroAng) > 0.01f) {
                juce::Path valArc;
                valArc.addCentredArc(cx, cy, 30.0f, 30.0f, 0.0f, std::min(zeroAng, curAng), std::max(zeroAng, curAng), true);
                g.setColour(colBubble);
                g.strokePath(valArc, juce::PathStrokeType(4.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            }

            // Outer Skirt
            g.setColour(juce::Colour(0xff2a0c20));
            g.fillEllipse(cx - 24.0f, cy - 24.0f, 48.0f, 48.0f);
            g.setColour(colEdge);
            g.drawEllipse(cx - 24.0f, cy - 24.0f, 48.0f, 48.0f, 1.2f);

            // Inner Matte Pink Cap
            juce::ColourGradient capGrad(colPastel, cx - 8.0f, cy - 10.0f, juce::Colour(0xff9e2b68), cx + 12.0f, cy + 14.0f, true);
            g.setGradientFill(capGrad);
            g.fillEllipse(cx - 18.0f, cy - 18.0f, 36.0f, 36.0f);
            g.setColour(colPastel.withAlpha(0.5f));
            g.drawEllipse(cx - 18.0f, cy - 18.0f, 36.0f, 36.0f, 1.0f);

            // Pointer Line
            float px0 = cx + 4.0f * std::sin(curAng), py0 = cy - 4.0f * std::cos(curAng);
            float px1 = cx + 15.0f * std::sin(curAng), py1 = cy - 15.0f * std::cos(curAng);
            g.setColour(colObsidian);
            g.drawLine(px0, py0, px1, py1, 3.2f);
            g.setColour(juce::Colours::white);
            g.drawLine(px0, py0, px1, py1, 1.4f);

            // Label
            g.setColour(juce::Colour(0xfff3c2dc));
            g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
            g.drawText(label, 0, 74, getWidth(), 14, juce::Justification::centred);

            // Value Readout Box (Flat Matte, No Text Shadow)
            juce::Rectangle<float> vb(cx - 35.0f, 90.0f, 70.0f, 18.0f);
            g.setColour(juce::Colour(0xff12050e));
            g.fillRoundedRectangle(vb, 5.0f);
            g.setColour(dragging ? colBubble : colEdge);
            g.drawRoundedRectangle(vb, 5.0f, 1.0f);
            g.setColour(colInk);
            g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 11.0f, juce::Font::bold));
            g.drawText(fmt(val), vb, juce::Justification::centred);
        }

        // 14% LESS SENSITIVE KNOB DRAG (0.00473 norm/px normal, 0.00103 Shift)
        void mouseDown(const juce::MouseEvent& e) override
        {
            if (!param) return;
            dragging = true;
            lastY = e.position.y;
            attachment->beginGesture();
            repaint();
        }

        void mouseDrag(const juce::MouseEvent& e) override
        {
            if (!dragging || !param) return;
            float dy = lastY - e.position.y;
            lastY = e.position.y;
            float sens = e.mods.isShiftDown() ? 0.00103f : 0.00473f;
            float nextNorm = juce::jlimit(0.0f, 1.0f, param->getValue() + dy * sens);
            attachment->setValueAsPartOfGesture( param->convertFrom0to1(nextNorm) );
        }

        void mouseUp(const juce::MouseEvent&) override
        {
            if (dragging) {
                dragging = false;
                if (param) attachment->endGesture();
                repaint();
            }
        }

        void mouseDoubleClick(const juce::MouseEvent&) override
        {
            if (param) attachment->setValueAsCompleteGesture(param->convertFrom0to1(param->getDefaultValue()));
        }

        void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
        {
            if (!param || std::abs(wheel.deltaY) < 1e-5f) return;
            float step = (wheel.deltaY > 0.0f ? 1.0f : -1.0f) * (e.mods.isShiftDown() ? 0.0034f : 0.0172f);
            float nextNorm = juce::jlimit(0.0f, 1.0f, param->getValue() + step);
            attachment->setValueAsCompleteGesture(param->convertFrom0to1(nextNorm));
        }

    private:
        juce::AudioProcessorValueTreeState& apvts;
        juce::String id, label;
        bool isBi = false, dimmed = false, dragging = false;
        float lastY = 0.0f;
        std::function<juce::String(float)> fmt;
        juce::RangedAudioParameter* param = nullptr;
        std::unique_ptr<juce::ParameterAttachment> attachment;
    };

    class ClashDollEditor : public juce::AudioProcessorEditor, private juce::Timer
    {
    public:
        ClashDollEditor(ClashDollProcessor& p) : AudioProcessorEditor(&p), proc(p)
        {
            setSize(1120, 680);

            auto addKnob = [this](const juce::String& id, const juce::String& lbl, bool bi, std::function<juce::String(float)> f) {
                auto k = std::make_unique<MatteKnob>(proc.apvts, id, lbl, bi, f);
                addAndMakeVisible(*k);
                knobs[id] = std::move(k);
            };

            auto pct = [](float v) { return juce::String((int)std::round(v * 100.0f)) + "%"; };
            auto sgn = [](float v) { return v >= 0.0f ? "+" : "-"; };

            addKnob("hz", "HZ SHIFT", true, [sgn](float v) {
                return sgn(v) + juce::String(std::abs(v), std::abs(v) < 10.0f ? 2 : 1) + " Hz";
            });
            addKnob("cents", "CENTS", true, [sgn](float v) { return sgn(v) + juce::String((int)std::abs(std::round(v))) + " ct"; });
            addKnob("grain", "GRAIN", false, [](float v) { return juce::String(v, 1) + " ms"; });
            addKnob("grind", "FM GRIND", false, pct);
            addKnob("dupe", "DUPE MIX", false, pct);
            addKnob("split", "STEREO SPLIT", false, [pct](float v) { return v < 0.005f ? juce::String("MONO") : pct(v); });
            addKnob("focus", "CLASH FOCUS", false, [](float v) { return "+" + juce::String(v * 18.0f, 1) + " dB"; });
            addKnob("drive", "DRIVE", false, [](float v) { return "+" + juce::String(v, 1) + " dB"; });
            addKnob("bias", "BIAS SPREAD", false, [](float v) { return "+/-" + juce::String(v, 2); });
            addKnob("sub", "SUB SAFE 120", false, [pct](float v) { return v < 0.005f ? juce::String("OFF") : pct(v); });
            addKnob("tone", "TONE TILT", true, [](float v) {
                return std::abs(v) < 0.01f ? juce::String("FLAT") : (v < 0.0f ? "DARK " : "BRIGHT ") + juce::String(std::abs(v) * 9.0f, 1);
            });
            addKnob("mix", "DRY / WET", false, pct);
            addKnob("out", "OUTPUT", false, [](float v) { return (v >= 0.0f ? "+" : "") + juce::String(v, 1) + " dB"; });

            startTimerHz(60);
        }

        void resized() override
        {
            // Module 01: Dissonance Engine
            knobs["hz"]->setBounds(40, 415, 86, 112);
            knobs["cents"]->setBounds(130, 415, 86, 112);
            knobs["grain"]->setBounds(220, 415, 86, 112);
            knobs["grind"]->setBounds(40, 528, 86, 112);

            // Module 02: Clash Mix
            knobs["dupe"]->setBounds(348, 372, 92, 112);
            knobs["split"]->setBounds(448, 372, 92, 112);
            knobs["focus"]->setBounds(348, 495, 92, 112);

            // Module 03: Spread Drive
            knobs["drive"]->setBounds(578, 462, 86, 112);
            knobs["bias"]->setBounds(668, 462, 86, 112);
            knobs["sub"]->setBounds(758, 462, 86, 112);

            // Module 04: Output
            knobs["tone"]->setBounds(884, 372, 92, 112);
            knobs["mix"]->setBounds(984, 372, 92, 112);
            knobs["out"]->setBounds(884, 488, 92, 112);
        }

        void timerCallback() override
        {
            int pm = (int)std::round(proc.apvts.getRawParameterValue("pitchMode")->load());
            knobs["hz"]->setDimmed(pm == 1);
            knobs["cents"]->setDimmed(pm != 1);
            knobs["grain"]->setDimmed(pm != 1);
            knobs["grind"]->setDimmed(pm != 2);
            repaint();
        }

        void mouseDown(const juce::MouseEvent& e) override
        {
            auto pt = e.position.toInt();
            for (int i = 0; i < 5; ++i)
                if (presetRects[i].contains(pt)) { applyPreset(i); return; }

            for (int i = 0; i < 3; ++i)
                if (pitchModeRects[i].contains(pt)) {
                    if (auto* p = proc.apvts.getParameter("pitchMode"))
                        p->setValueNotifyingHost(p->convertTo0to1((float)i));
                    activePreset = -1;
                    return;
                }

            for (int i = 0; i < 4; ++i)
                if (distModeRects[i].contains(pt)) {
                    if (auto* p = proc.apvts.getParameter("distMode"))
                        p->setValueNotifyingHost(p->convertTo0to1((float)i));
                    activePreset = -1;
                    return;
                }
        }

        void applyPreset(int idx)
        {
            activePreset = idx;
            const auto& pr = PRESETS[idx];
            auto setP = [this](const char* id, float val) {
                if (auto* p = proc.apvts.getParameter(id))
                    p->setValueNotifyingHost(p->convertTo0to1(val));
            };
            setP("pitchMode", (float)pr.pitchMode);
            setP("hz", pr.hz); setP("cents", pr.cents); setP("grain", pr.grain); setP("grind", pr.grind);
            setP("dupe", pr.dupe); setP("split", pr.split); setP("focus", pr.focus);
            setP("drive", pr.drive); setP("bias", pr.bias); setP("distMode", (float)pr.distMode);
            setP("sub", pr.sub); setP("tone", pr.tone); setP("mix", pr.mix); setP("out", pr.out);
        }

        void drawPlate(juce::Graphics& g, juce::Rectangle<float> r, const juce::String& num, const juce::String& title)
        {
            juce::ColourGradient grad(colPanel2, r.getX(), r.getY(), colPanel, r.getX(), r.getBottom(), false);
            g.setGradientFill(grad);
            g.fillRoundedRectangle(r, 16.0f);
            g.setColour(colEdge);
            g.drawRoundedRectangle(r, 16.0f, 1.2f);

            juce::Rectangle<float> badge(r.getX() + 14.0f, r.getY() + 12.0f, 22.0f, 15.0f);
            g.setColour(colBubble);
            g.fillRoundedRectangle(badge, 4.0f);
            g.setColour(colObsidian);
            g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::bold));
            g.drawText(num, badge, juce::Justification::centred);

            g.setColour(colBubble);
            g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
            g.drawText(title, (int)r.getX() + 42, (int)r.getY() + 12, 180, 15, juce::Justification::centredLeft);
        }

        void drawSegButton(juce::Graphics& g, juce::Rectangle<int> r, const juce::String& text, bool on)
        {
            auto rf = r.toFloat();
            g.setColour(on ? juce::Colour(0xff4c1539) : juce::Colour(0xff260e1f));
            g.fillRoundedRectangle(rf, 8.0f);
            g.setColour(on ? colMagenta : colEdge);
            g.drawRoundedRectangle(rf, 8.0f, 1.2f);

            g.setColour(on ? colBubble : juce::Colour(0xff4a1835));
            g.fillEllipse(rf.getX() + 8.0f, rf.getCentreY() - 3.5f, 7.0f, 7.0f);

            g.setColour(on ? juce::Colours::white : juce::Colour(0xffe8b9d2));
            g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
            g.drawText(text, r.withTrimmedLeft(18), juce::Justification::centred);
        }

        void paint(juce::Graphics& g) override
        {
            g.fillAll(colObsidian);

            // Outer Hardware Chassis (Matte Satin Finish, Zero Glow)
            juce::Rectangle<float> rig(12.0f, 12.0f, getWidth() - 24.0f, getHeight() - 24.0f);
            juce::ColourGradient rigGrad(juce::Colour(0xff321329), 0, 0, juce::Colour(0xff1c0b17), 0, (float)getHeight(), false);
            g.setGradientFill(rigGrad);
            g.fillRoundedRectangle(rig, 22.0f);
            g.setColour(colBubble.withAlpha(0.45f));
            g.drawRoundedRectangle(rig, 22.0f, 2.0f);

            // Corner Screws
            for (auto pt : { juce::Point<float>(26, 26), juce::Point<float>(getWidth() - 26.0f, 26),
                             juce::Point<float>(26, getHeight() - 26.0f), juce::Point<float>(getWidth() - 26.0f, getHeight() - 26.0f) }) {
                g.setColour(colBubble);
                g.fillEllipse(pt.x - 6.0f, pt.y - 6.0f, 12.0f, 12.0f);
                g.setColour(colEdge);
                g.drawLine(pt.x - 3.5f, pt.y - 2.5f, pt.x + 3.5f, pt.y + 2.5f, 1.5f);
            }

            // Brand Header
            juce::Rectangle<float> logoBox(34.0f, 26.0f, 40.0f, 40.0f);
            g.setColour(colMagenta);
            g.fillRoundedRectangle(logoBox, 11.0f);
            g.setColour(juce::Colours::white);
            g.setFont(juce::FontOptions(20.0f, juce::Font::bold));
            g.drawText("*", logoBox, juce::Justification::centred);

            g.setColour(colInk);
            g.setFont(juce::FontOptions(21.0f, juce::Font::bold));
            g.drawText("CLASH-DOLL NPU-1", 86, 25, 260, 24, juce::Justification::centredLeft);
            g.setColour(colMuted);
            g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain));
            g.drawText("DUAL-VOICE DISSONANCE / BIAS SPREAD", 86, 48, 280, 14, juce::Justification::centredLeft);

            // 5 Preset Pills
            int px = 385;
            for (int i = 0; i < 5; ++i) {
                int w = (i == 0 || i == 4) ? 142 : 132;
                presetRects[i] = juce::Rectangle<int>(px, 31, w, 28);
                bool on = (activePreset == i);
                g.setColour(on ? colBubble : juce::Colour(0xff2e1325));
                g.fillRoundedRectangle(presetRects[i].toFloat(), 14.0f);
                g.setColour(on ? juce::Colours::white : colEdge);
                g.drawRoundedRectangle(presetRects[i].toFloat(), 14.0f, 1.0f);
                g.setColour(on ? colObsidian : colPastel);
                g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
                g.drawText(PRESETS[i].name, presetRects[i], juce::Justification::centred);
                px += w + 8;
            }

            // Screens: Stereo Scope & Goniometer
            juce::Rectangle<float> scopeBox(28.0f, 80.0f, 756.0f, 232.0f);
            juce::Rectangle<float> vecBox(796.0f, 80.0f, 296.0f, 232.0f);
            g.setColour(juce::Colour(0xff12050e));
            g.fillRoundedRectangle(scopeBox, 14.0f);
            g.fillRoundedRectangle(vecBox, 14.0f);
            g.setColour(colEdge);
            g.drawRoundedRectangle(scopeBox, 14.0f, 1.2f);
            g.drawRoundedRectangle(vecBox, 14.0f, 1.2f);

            // Scope Grid
            g.setColour(colBubble.withAlpha(0.08f));
            for (int x = 1; x < 12; ++x)
                g.drawVerticalLine((int)(scopeBox.getX() + x * scopeBox.getWidth() / 12.0f), scopeBox.getY(), scopeBox.getBottom());
            for (int y = 1; y < 6; ++y)
                g.drawHorizontalLine((int)(scopeBox.getY() + y * scopeBox.getHeight() / 6.0f), scopeBox.getX(), scopeBox.getRight());
            g.setColour(colBubble.withAlpha(0.22f));
            g.drawHorizontalLine((int)scopeBox.getCentreY(), scopeBox.getX(), scopeBox.getRight());

            // Fetch Audio Ring Buffer
            int wp = proc.visWritePos.load(std::memory_order_relaxed);
            std::array<float, 1024> bL{}, bR{};
            for (int i = 0; i < 1024; ++i) {
                int idx = (wp - 1024 + i + ClashDollProcessor::kScopeSize) & (ClashDollProcessor::kScopeSize - 1);
                bL[i] = proc.visL[idx];
                bR[i] = proc.visR[idx];
            }

            int trig = 0;
            for (int i = 1; i < 512; ++i)
                if (bL[i - 1] < 0.0f && bL[i] >= 0.0f) { trig = i; break; }

            juce::Path pathL, pathR;
            const int drawN = 512;
            for (int i = 0; i < drawN; ++i) {
                float x = scopeBox.getX() + 8.0f + (i / (float)(drawN - 1)) * (scopeBox.getWidth() - 16.0f);
                float yL = scopeBox.getCentreY() - juce::jlimit(-1.0f, 1.0f, bL[trig + i]) * (scopeBox.getHeight() * 0.42f);
                float yR = scopeBox.getCentreY() - juce::jlimit(-1.0f, 1.0f, bR[trig + i]) * (scopeBox.getHeight() * 0.42f);
                if (i == 0) { pathL.startNewSubPath(x, yL); pathR.startNewSubPath(x, yR); }
                else        { pathL.lineTo(x, yL);          pathR.lineTo(x, yR); }
            }
            g.setColour(colInk.withAlpha(0.85f));
            g.strokePath(pathR, juce::PathStrokeType(1.5f));
            g.setColour(colMagenta);
            g.strokePath(pathL, juce::PathStrokeType(1.8f));

            // Goniometer
            float vcx = vecBox.getCentreX(), vcy = vecBox.getCentreY() - 6.0f, vrad = 86.0f;
            g.setColour(colBubble.withAlpha(0.15f));
            g.drawEllipse(vcx - vrad, vcy - vrad, vrad * 2.0f, vrad * 2.0f, 1.0f);
            g.drawLine(vcx - vrad, vcy, vcx + vrad, vcy, 1.0f);
            g.drawLine(vcx, vcy - vrad, vcx, vcy + vrad, 1.0f);

            juce::Path vecPath;
            double sl = 0.0, sr = 0.0, slr = 0.0;
            for (int i = 0; i < 1024; ++i) {
                float l = bL[i], r = bR[i];
                sl += l * l; sr += r * r; slr += l * r;
                float vx = vcx + (r - l) * 0.7071f * vrad;
                float vy = vcy - (l + r) * 0.7071f * vrad;
                if (i == 0) vecPath.startNewSubPath(vx, vy);
                else        vecPath.lineTo(vx, vy);
            }
            g.setColour(colBubble.withAlpha(0.75f));
            g.strokePath(vecPath, juce::PathStrokeType(1.2f));

            float corr = (sl > 1e-9 && sr > 1e-9) ? (float)(slr / std::sqrt(sl * sr)) : 1.0f;
            corrSmooth += (corr - corrSmooth) * 0.15f;

            // Correlation Bar
            juce::Rectangle<float> corrBar(vecBox.getX() + 14.0f, vecBox.getBottom() - 18.0f, vecBox.getWidth() - 28.0f, 7.0f);
            g.setColour(juce::Colour(0xff2a0c20));
            g.fillRoundedRectangle(corrBar, 3.5f);
            g.setColour(colEdge);
            g.drawRoundedRectangle(corrBar, 3.5f, 1.0f);
            float dotX = corrBar.getX() + ((corrSmooth + 1.0f) * 0.5f) * corrBar.getWidth();
            g.setColour(juce::Colours::white);
            g.fillRoundedRectangle(dotX - 2.0f, corrBar.getY() - 2.0f, 4.0f, 11.0f, 2.0f);

            // Screen HUD Text
            g.setColour(colBubble);
            g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::bold));
            g.drawText("STEREO SCOPE  [L: MAGENTA / R: WHITE]", (int)scopeBox.getX() + 14, (int)scopeBox.getY() + 8, 280, 14, juce::Justification::centredLeft);

            float pkL = proc.meterPeakL.load(std::memory_order_relaxed);
            float pkR = proc.meterPeakR.load(std::memory_order_relaxed);
            auto dbStr = [](float v) { return v < 1e-4f ? juce::String("-inf") : juce::String(juce::Decibels::gainToDecibels(v), 1); };
            g.drawText("PK L " + dbStr(pkL) + " / R " + dbStr(pkR) + " dB", (int)scopeBox.getRight() - 210, (int)scopeBox.getY() + 8, 196, 14, juce::Justification::centredRight);
            g.drawText("GONIOMETER", (int)vecBox.getX() + 14, (int)vecBox.getY() + 8, 110, 14, juce::Justification::centredLeft);
            g.drawText(juce::String("CORR ") + (corrSmooth >= 0.0f ? "+" : "") + juce::String(corrSmooth, 2),
                       (int)vecBox.getRight() - 110, (int)vecBox.getY() + 8, 96, 14, juce::Justification::centredRight);

            // 4 Bottom Module Plates
            drawPlate(g, { 28.0f,  324.0f, 294.0f, 304.0f }, "01", "DISSONANCE ENGINE");
            drawPlate(g, { 334.0f, 324.0f, 222.0f, 304.0f }, "02", "CLASH MIX");
            drawPlate(g, { 568.0f, 324.0f, 290.0f, 304.0f }, "03", "SPREAD DRIVE (4x OS)");
            drawPlate(g, { 870.0f, 324.0f, 222.0f, 304.0f }, "04", "OUTPUT");

            int pm = (int)std::round(proc.apvts.getRawParameterValue("pitchMode")->load());
            const char* pmNames[3] = { "HZ SHIFT", "CENTS", "FM GRIND" };
            for (int i = 0; i < 3; ++i) {
                pitchModeRects[i] = juce::Rectangle<int>(42 + i * 90, 358, 84, 34);
                drawSegButton(g, pitchModeRects[i], pmNames[i], pm == i);
            }

            int dm = (int)std::round(proc.apvts.getRawParameterValue("distMode")->load());
            const char* dmNames[4] = { "BRUTAL CLIP", "PLASMA FOLD", "DIODE CRUNCH", "QUANTIZE RIPPER" };
            for (int i = 0; i < 3; ++i) {
                distModeRects[i] = juce::Rectangle<int>(582 + i * 90, 358, 84, 34);
                drawSegButton(g, distModeRects[i], dmNames[i], dm == i);
            }
            distModeRects[3] = juce::Rectangle<int>(582, 400, 264, 30);
            drawSegButton(g, distModeRects[3], dmNames[3], dm == 3);

            // Output VU Meters & Limiter LED
            bool limActive = proc.meterGr.load(std::memory_order_relaxed) < 0.97f;
            g.setColour(limActive ? colMagenta : juce::Colour(0xff4a1835));
            g.fillEllipse(1066.0f, 335.0f, 10.0f, 10.0f);

            auto drawVu = [&](float y, float pk, const char* lbl) {
                g.setColour(colMuted);
                g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::bold));
                g.drawText(lbl, 886, (int)y - 2, 14, 12, juce::Justification::centredLeft);
                juce::Rectangle<float> bar(902.0f, y, 174.0f, 8.0f);
                g.setColour(juce::Colour(0xff12050e));
                g.fillRoundedRectangle(bar, 4.0f);
                g.setColour(colEdge);
                g.drawRoundedRectangle(bar, 4.0f, 1.0f);
                float norm = juce::jlimit(0.0f, 1.0f, (juce::Decibels::gainToDecibels(pk + 1e-6f) + 48.0f) / 48.0f);
                if (norm > 0.01f) {
                    g.setColour(colBubble);
                    g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * norm), 4.0f);
                }
            };
            drawVu(596.0f, pkL, "L");
            drawVu(610.0f, pkR, "R");

            // Bottom Signal Chain Strip
            g.setColour(colMuted);
            g.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::bold));
            juce::String chain = "IN  >  HILBERT I/Q  >  [V2: " + juce::String(pmNames[pm]) + "]  >  SPLIT  >  FOCUS  >  DRIVE +/-BIAS  >  [4x " + juce::String(dmNames[dm]) + "]  >  DC 15Hz  >  SUB SAFE 120Hz  >  TILT  >  LIMIT -0.1dB";
            g.drawText(chain, 28, 636, getWidth() - 56, 20, juce::Justification::centred);
        }

    private:
        ClashDollProcessor& proc;
        std::map<juce::String, std::unique_ptr<MatteKnob>> knobs;
        juce::Rectangle<int> presetRects[5], pitchModeRects[3], distModeRects[4];
        int activePreset = 0;
        float corrSmooth = 1.0f;
    };
}

juce::AudioProcessorEditor* ClashDollProcessor::createEditor() { return new ClashDollEditor(*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new ClashDollProcessor(); }
