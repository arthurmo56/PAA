#include "descritor/imagedescriptor.hpp"
#include "estruturas/mtree.hpp"
#include "estruturas/hashLSH.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <ctime>
namespace fs = std::filesystem;

// --- Função auxiliar: garante carregamento limpo da imagem ---
cv::Mat loadImage(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Erro: não foi possível abrir " << path << std::endl;
        return {};
    }
    if (img.channels() == 4)
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    cv::resize(img, img, cv::Size(256, 256));
    return img;
}

// --- Busca com Mtree ---
void runMtree(const std::string& queryPath, int K) {
    Rect boundary{0.5f, 0.5f, 0.5f, 0.5f};
    Quadtree qt(boundary, 4);

    for (const auto& entry : fs::directory_iterator("images")) {
        if (entry.is_regular_file()) {
            cv::Mat img = loadImage(entry.path().string());
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, entry.path().string(), d.hist, d.muH, d.muS};
            Point p{d.muH, d.muS, r};
            qt.insert(p);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record rq{0, queryPath, dq.hist, dq.muH, dq.muS};
    Point qpoint{dq.muH, dq.muS, rq};

    float radius = 0.2f; // aumentar raio para pegar mais candidatos
    Rect range{qpoint.x, qpoint.y, radius, radius};
    std::vector<Point> candidates;
    qt.queryRange(range, candidates);

    std::vector<std::pair<float, Record>> dists;
    for (auto& p : candidates) {
        float d = chi2_distance(rq.hist, p.record.hist);
        dists.push_back({d, p.record});
    }
    std::sort(dists.begin(), dists.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    std::cout << "Resultados (Quadtree):\n";
    for (int i = 0; i < K && i < (int)dists.size(); i++) {
        std::cout << dists[i].second.filepath << "\n";
    }
}

// --- Busca com Hash LSH ---
void runHashLSH(const std::string& queryPath, int K) {
    int dimensao = 8*9 + 2; // Histograma de 8*3*3 + muH + muS.

    // parâmetros do LSH (K funções, L tabelas, w largura do bucket)
    int k = 6;
    int L = 10;
    float w = 0.6f;

    LSHIndex hindex(dimensao, k, L, w);

    for (const auto& entry : fs::directory_iterator("images")) {
        if (entry.is_regular_file()) {
            cv::Mat img = loadImage(entry.path().string());
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, entry.path().string(), d.hist, d.muH, d.muS};
            hindex.add(r, d);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record rq{0, queryPath, dq.hist, dq.muH, dq.muS};

    auto results = hindex.query(rq, dq, K);

    std::cout << "Resultados (Hash):\n";
    for (auto& r : results) {
        std::cout << r.filepath << "\n";
    }

    std::cout << "Comparações: "<< hindex.getComparisonCount() << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <imagem_query>" << std::endl;
        return -1;
    }

    std::string queryPath = argv[1];
    int K = 5;

    std::cout << "Escolha o método:\n";
    std::cout << "1 - Mtree\n";
    std::cout << "2 - Hash com LSH\n";
    int choice;
    std::cin >> choice;

    clock_t start = clock();

    if (choice == 1) {
        runMtree(queryPath, K);
    } else if (choice == 2) {
        runHashLSH(queryPath, K);
    } else {
        std::cout << "Opção inválida.\n";
    }

    clock_t end = clock();
    double elapsed = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tempo de execução: " << elapsed << " segundos\n";

    return 0;
}
