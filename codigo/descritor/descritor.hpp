#pragma once
#include <string>
#include <vector>

struct Record {
    int id;                     // identificador da imagem
    std::string filepath;       // caminho da imagem
    std::vector<float> hist;    // descritor (72 bins normalizado)
    float muH;                  // média Hue
    float muS;                  // média Saturação
};
