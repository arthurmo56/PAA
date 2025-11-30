#pragma once
#include "../descritor/imagedescriptor.hpp"
#include "../descritor/descritor.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <sstream>

struct LSHFunction {
    std::vector<float> a; // Vetor de projeção aleatória
    float b; // Offset aleatório
    float w; // Largura do bucket

    LSHFunction(int dimension, float w_param) : w(w_param) {
        std::random_device rd;
        std::mt19937 gen(rd());
        // distribuição normal para o vetor de projeção 'a'
        std::normal_distribution<> d_a(0, 1);
        // distribuição uniforme para o offset 'b'
        std::uniform_real_distribution<> d_b(0.0, w);

        a.resize(dimension);
        for (int i = 0; i < dimension; ++i) {
            a[i] = d_a(gen);
        }
        b = d_b(gen);
    }

    // calcula o valor do hash para um vetor de entrada (v)
    int computeHash(const std::vector<float>& v) const {
        float projection = 0.0f;
        for (size_t i = 0; i < v.size(); ++i) {
            projection += a[i] * v[i];
        }
        // retorna a função h(v) = floor((a · v + b) / w)
        return std::floor((projection + b) / w);
    }
};

// estrutura para uma única Tabela Hash LSH
struct LSHHashTable {
    // K funções hash por tabela
    std::vector<LSHFunction> hashFunctions;
    // armazena as chaves de hash combinadas e os registros
    std::unordered_map<std::string, std::vector<Record>> table;

    LSHHashTable(int dimension, int K, float w_param) {
        for (int i = 0; i < K; ++i) {
            hashFunctions.emplace_back(dimension, w_param);
        }
    }

    // função que cria o vetor de característica completo
    std::vector<float> createFeatureVector(const Descriptor& d) const {
        std::vector<float> v = d.hist; // Histograma
        v.push_back(d.muH); // Média Hue
        v.push_back(d.muS); // Média Saturation

        return v;
    }

    std::string makeLSHKey(const std::vector<float>& v) const {
        std::stringstream ss;
        for (const auto& hf : hashFunctions) {
            ss << hf.computeHash(v) << "_";
        }
        return ss.str();
    }
};

class LSHIndex {
private:
    std::vector<LSHHashTable> hashTables;
    int dimension;
    int K;
    int L;
    float w;

public:
    long long comparisonCount = 0;
    LSHIndex(int dim, int k_param, int l_param, float w_param = 1.0f)
        : dimension(dim), K(k_param), L(l_param), w(w_param) {
        for (int i = 0; i < L; ++i) {
            hashTables.emplace_back(dimension, K, w);
        }
    }

    long long getComparisonCount() const { return comparisonCount; }

    void resetComparisonCount() { comparisonCount = 0; }

    void add(const Record& rec, const Descriptor& d) {
        for (auto& ht : hashTables) {
            // cria o vetor de alta dimensão
            std::vector<float> featureVector = ht.createFeatureVector(d);
            // calcula a chave para a tabela LSH
            std::string key = ht.makeLSHKey(featureVector);
            // adiciona o registro
            ht.table[key].push_back(rec);
        }
    }

    std::vector<Record> query(const Record& q, const Descriptor& d, int K_nn) {
        std::vector<Record> candidates;
        std::unordered_map<std::string, bool> alreadyChecked;

       // geração de candidatos pra analise
        std::vector<float> featureVector = hashTables[0].createFeatureVector(d);
        for (auto& ht : hashTables) {
            std::string key = ht.makeLSHKey(featureVector);
            if (ht.table.count(key)) {
                for (const auto& rec : ht.table[key]) {
                    if (alreadyChecked.find(rec.filepath) == alreadyChecked.end()) {
                        candidates.push_back(rec);
                        alreadyChecked[rec.filepath] = true;
                    }
                }
            }
        }

        // refinamento com o chi_distance pro calculo euclidiano
        std::vector<std::pair<float, Record>> dists;
        for (auto& r : candidates) {
            float dist = chi2_distance(q.hist, r.hist); 
            dists.push_back({dist, r});
            comparisonCount++;
        }
        std::sort(dists.begin(), dists.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });

        // retorna os K-vizinhos mais próximos
        std::vector<Record> result;
        for (int i = 0; i < K_nn && i < (int)dists.size(); i++) {
            result.push_back(dists[i].second);
        }
        return result;
    }
};