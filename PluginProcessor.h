#pragma once
#include <JuceHeader.h>
#include <array>
#include <vector>
#include <atomic>
#include <cmath>

class ClashDollProcessor : public juce::AudioProcessor
{
public:
    ClashDollProcessor();
    ~ClashDollProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "CLASH-DOLL NPU-1"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Lock-free visual data for 60fps Scope & Goniometer
    static constexpr int kScopeSize = 2048;
    std::array<float, kScopeSize> visL {}, visR {};
    std::atomic<int> visWritePos { 0 };
    std::atomic<float> meterPeakL { 0.0f }, meterPeakR { 0.0f }, meterGr { 1.0f };

private:
    static constexpr double TWO_PI = 6.28318530717958647692;
    static constexpr double ANTI_DENORMAL = 1e-20;

    struct HilbertIQ {
        double ci[4]{}, cq[4]{};
        double mi[16]{}, mq[16]{};
        double prev = 0.0, i = 0.0, q = 0.0;
        void init(double fs);
        static double runStage(double x, const double* c, double* m);
        void process(double x);
    };

    struct Upsampler2x {
        std::vector<double> c, x, y;
        void init(const double* coefs, int n) {
            c.assign(coefs, coefs + n);
            x.assign(n, 0.0);
            y.assign(n, 0.0);
        }
        void process(double input, double* out, int offset) {
            int n = (int)c.size();
            double e = input, o = input;
            for (int i = 0; i < n; i += 2) {
                double ye = c[i] * (e - y[i]) + x[i];
                x[i] = e; y[i] = ye; e = ye;
                if (i + 1 < n) {
                    double yo = c[i + 1] * (o - y[i + 1]) + x[i + 1];
                    x[i + 1] = o; y[i + 1] = yo; o = yo;
                }
            }
            out[offset] = e;
            out[offset + 1] = o;
        }
    };

    struct Downsampler2x {
        std::vector<double> c, x, y;
        void init(const double* coefs, int n) {
            c.assign(coefs, coefs + n);
            x.assign(n, 0.0);
            y.assign(n, 0.0);
        }
        double process(double first, double second) {
            int n = (int)c.size();
            double e = second, o = first;
            for (int i = 0; i < n; i += 2) {
                double ye = c[i] * (e - y[i]) + x[i];
                x[i] = e; y[i] = ye; e = ye;
                if (i + 1 < n) {
                    double yo = c[i + 1] * (o - y[i + 1]) + x[i + 1];
                    x[i + 1] = o; y[i + 1] = yo; o = yo;
                }
            }
            return 0.5 * (e + o);
        }
    };

    struct Biquad {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0, z1 = 0.0, z2 = 0.0;
        void reset() { z1 = z2 = 0.0; }
        void setPeak(double fs, double f0, double q, double gainDb) {
            double A = std::pow(10.0, gainDb / 40.0);
            double w0 = TWO_PI * f0 / fs;
            double cs = std::cos(w0), al = std::sin(w0) / (2.0 * q);
            double a0 = 1.0 + al / A;
            b0 = (1.0 + al * A) / a0; b1 = (-2.0 * cs) / a0; b2 = (1.0 - al * A) / a0;
            a1 = (-2.0 * cs) / a0; a2 = (1.0 - al / A) / a0;
        }
        void setLowpass(double fs, double f0, double q) {
            double w0 = TWO_PI * f0 / fs;
            double cs = std::cos(w0), al = std::sin(w0) / (2.0 * q);
            double a0 = 1.0 + al;
            b0 = ((1.0 - cs) * 0.5) / a0; b1 = (1.0 - cs) / a0; b2 = b0;
            a1 = (-2.0 * cs) / a0; a2 = (1.0 - al) / a0;
        }
        void setHighpass(double fs, double f0, double q) {
            double w0 = TWO_PI * f0 / fs;
            double cs = std::cos(w0), al = std::sin(w0) / (2.0 * q);
            double a0 = 1.0 + al;
            b0 = ((1.0 + cs) * 0.5) / a0; b1 = -(1.0 + cs) / a0; b2 = b0;
            a1 = (-2.0 * cs) / a0; a2 = (1.0 - al) / a0;
        }
        double process(double x) {
            double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    enum SmoothIdx {
        S_HZ = 0, S_CENTS, S_GRAIN, S_GRIND, S_DUPE, S_SPLIT, S_FOCUS, S_DRIVE, S_BIAS,
        S_SUB, S_TONE, S_MIX, S_OUT, S_PM0, S_PM1, S_PM2, S_DM0, S_DM1, S_DM2, S_DM3,
        SMOOTH_COUNT
    };

    double fs = 48000.0;
    std::array<double, SMOOTH_COUNT> s{}, t{};

    HilbertIQ hil[2];
    double theta = 0.0;

    int dlSize = 16384;
    int dlMask = 16383;
    std::vector<float> dl[2];
    int dlW = 0;
    double psPhase = 0.0;

    Biquad pre[2], post[2];
    double lastFocusDb = -100.0;

    Upsampler2x up1[2], up2[2];
    Downsampler2x dn2[2], dn1[2];
    double dcR4 = 0.999, dcX[2]{}, dcY[2]{};

    double dcR1 = 0.998, cleanDcX = 0.0, cleanDcY = 0.0;
    Biquad wetLp[2][2], wetHp[2][2], cleanLp[2];

    double tiltA = 0.01, tiltLp[2]{}, tiltLow = 1.0, tiltHigh = 1.0;
    double limEnv = 1.0, limRel = 0.001;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ClashDollProcessor)
};
