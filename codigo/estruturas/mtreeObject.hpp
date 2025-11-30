#ifndef MTREEOBJECT_HPP
#define MTREEOBJECT_HPP

#include "../descritor/imagedescriptor.hpp"
#include <cmath>
#include <string>

using namespace std;

// Wrapper para armazenar no M-Tree
struct MTreeObject {
    Descriptor desc;
    int id;            // identificador da imagem / objeto
    string filepath;  // opcional (caminho, etc.)

    MTreeObject() = default;
    MTreeObject(int id_, const Descriptor& d, const string& m = "")
        : desc(d), id(id_), filepath(m) {}
};

// distância circular entre valores em [0,1]
inline float circular_dist(float a, float b) {
    float diff = fabs(a - b);
    return min(diff, 1.0f - diff);
}

// função de distância combinada (métrica usada pela M-Tree)
// Combina Chi^2(hist), distância circular do hue médio (muH) e diferença de saturação média (muS).
inline float mtree_distance(const MTreeObject& A, const MTreeObject& B) {
    float d_hist = chi2_distance(A.desc.hist, B.desc.hist);
    float d_hue  = circular_dist(A.desc.muH, B.desc.muH);
    float d_sat  = fabs(A.desc.muS - B.desc.muS);

    // Pesos ajustáveis
    const float W_HIST = 1.0f;
    const float W_HUE  = 0.5f;
    const float W_SAT  = 0.25f;

    return W_HIST * d_hist + W_HUE * d_hue + W_SAT * d_sat;
}

#endif // MTREEOBJECT_HPP
