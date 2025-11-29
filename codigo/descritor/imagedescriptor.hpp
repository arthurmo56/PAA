#ifndef IMAGEDESCRIPTOR_HPP
#define IMAGEDESCRIPTOR_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Configuração da quantização
const int H_BINS = 8;   // Hue dividido em 8
const int S_BINS = 3;   // Saturation em 3
const int V_BINS = 3;   // Value em 3
const int HIST_SIZE = H_BINS * S_BINS * V_BINS;

struct Descriptor {
    std::vector<float> hist; // histograma normalizado
    float muH;               // média circular do Hue
    float muS;               // média da Saturação
};

// Função para gerar o descritor
Descriptor computeDescriptor(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<float> hist(HIST_SIZE, 0.0f);

    double sumSin = 0.0, sumCos = 0.0;
    double sumS = 0.0;
    int nPixels = hsv.rows * hsv.cols;

    for (int r = 0; r < hsv.rows; r++) {
        for (int c = 0; c < hsv.cols; c++) {
            cv::Vec3b pix = hsv.at<cv::Vec3b>(r, c);
            int h = pix[0]; // Hue [0,179]
            int s = pix[1]; // Saturation [0,255]
            int v = pix[2]; // Value [0,255]

            // Quantização
            int hBin = (h * H_BINS) / 180;   // Hue
            int sBin = (s * S_BINS) / 256;   // Saturation
            int vBin = (v * V_BINS) / 256;   // Value

            int idx = hBin * (S_BINS * V_BINS) + sBin * V_BINS + vBin;
            hist[idx] += 1.0f;

            // Médias
            double angle = (h / 180.0) * 2.0 * M_PI; // Hue em radianos
            sumCos += cos(angle);
            sumSin += sin(angle);
            sumS   += s / 255.0;
        }
    }

    // Normaliza histograma
    for (auto& v : hist) v /= nPixels;

    // Média circular para Hue
    float muH = atan2(sumSin, sumCos); // [-pi, pi]
    if (muH < 0) muH += 2 * M_PI;
    muH /= 2 * M_PI; // normaliza para [0,1]

    // Média da Saturação
    float muS = sumS / nPixels;

    return {hist, muH, muS};
}

inline float chi2_distance(const std::vector<float>& hist1, const std::vector<float>& hist2) {
    float dist = 0.0f;
    // Assumindo que os vetores têm o mesmo tamanho e que o tamanho é > 0
    for (size_t i = 0; i < hist1.size(); ++i) {
        float num = hist1[i] - hist2[i];
        float den = hist1[i] + hist2[i];
        
        // Evita divisão por zero e NaN. Se a soma dos bins for 0, a contribuição é 0.
        if (den > 1e-6) {
            dist += (num * num) / den;
        }
    }
    // A distância Chi-Quadrado tem um fator de 1/2.
    return 0.5f * dist;
}

#endif // IMAGEDESCRIPTOR_HPP