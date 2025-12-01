#ifndef MTREE_HPP
#define MTREE_HPP

#include "MTreeObject.hpp"
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <queue>
#include <functional>
#include <optional>
#include <cassert>
#include <iostream>
#include <atomic>
#define MTREE_DEBUG 0

using namespace std;

// Implementação de M-Tree (estrutura métrica para buscas por similaridade)
// Conceitos principais:
// - Entry: entrada de nó (pode ser objeto folha ou apontar para um filho).
// - Node: nó da árvore (folha ou interno) contendo várias entries.
// - Inserção: escolhe subárvore com menor expansão; divide nó ao ultrapassar M.
// - Consulta: range query e k-NN com poda por limites inferiores (LB).

class MTree
{
public:
    struct Node;
    struct Entry
    {
        // Objeto usado como pivô no nó
        MTreeObject routingObj;

        // Ponteiro para nó filho ou nulo (folha)
        shared_ptr<Node> child = nullptr;

        // Raio de cobertura
        float radius = 0.0f;

        // Distância ao pivô do nó pai (0 para a raiz)
        float distToParent = 0.0f;

        // child == nullptr indica entrada de folha, routingObj é o próprio objeto
        bool isLeafEntry() const { return child == nullptr; }
    };

    struct Node
    {
        bool isLeaf = true;
        vector<Entry> entries;
        weak_ptr<Node> parent;

        Node(bool leaf = true) : isLeaf(leaf) {}
    };

    // Parâmetros do nó
    size_t M = 50; // capacidade máxima
    size_t m = 25; // preenchimento mínimo (≈ M/2)

    shared_ptr<Node> root;

    // Instrumentação
    static atomic<size_t> NODE_INSTANCES; // nós criados
    static atomic<size_t> SPLIT_COUNT;    // splits realizados

    MTree(size_t maxEntries = 50)
    {
        M = maxEntries;
        m = max((size_t)2, maxEntries / 2);
        root = make_shared<Node>(true);
        NODE_INSTANCES.fetch_add(1, memory_order_relaxed);
    }

    void insert(const MTreeObject &obj)
    {
        insertRecursive(root, obj);
    }

    // Range query: retorna todos objetos com distância <= radius
    vector<MTreeObject> rangeQuery(const MTreeObject &query, float radius)
    {
        vector<MTreeObject> results;
        rangeRecursive(root, query, radius, results);
        return results;
    }

    // k-NN (retorna até k objetos mais próximos)
    vector<pair<MTreeObject, float>> knn(const MTreeObject &query, size_t k)
    {
        PQ pq;

        using ResultElem = pair<float, MTreeObject>;
        struct ResCmp
        {
            bool operator()(const ResultElem &a, const ResultElem &b) const { return a.first < b.first; }
        };
        priority_queue<ResultElem, vector<ResultElem>, ResCmp> results;

        pushNodeEntriesToPQ(root, query, pq);

        while (!pq.empty())
        {
            auto [lb, nodeAndIdx] = pq.top();
            pq.pop();

            auto node = nodeAndIdx.first;
            size_t idx = nodeAndIdx.second;
            const Entry &e = node->entries[idx];

            if (!e.isLeafEntry())
            {
                pushNodeEntriesToPQ(e.child, query, pq);
            }
            else
            {
                float d = safe_dist(query, e.routingObj);

                if (results.size() < k)
                {
                    results.emplace(d, e.routingObj);
                }
                else if (d < results.top().first)
                {
                    results.pop();
                    results.emplace(d, e.routingObj);
                }
            }

            if (results.size() == k && !pq.empty())
            {
                float worst = results.top().first;
                float nextLB = pq.top().first;
                if (nextLB >= worst)
                    break;
            }
        }

        vector<pair<MTreeObject, float>> out;
        while (!results.empty())
        {
            out.emplace_back(results.top().second, results.top().first);
            results.pop();
        }
        reverse(out.begin(), out.end());
        return out;
    }

    // Remoção
    bool remove(int objectId)
    {
        auto [node, idx] = findLeafEntry(root, objectId);
        if (!node)
            return false;
        // remove entry
        MTreeObject orphan = node->entries[idx].routingObj;
        node->entries.erase(node->entries.begin() + idx);

        handleUnderflow(node);
        return true;
    }

    // Estatísticas para debug
    void debug_print_stats()
    {
        int height = computeHeight(root);
        size_t nodes = 0, leaves = 0, totalEntries = 0;
        collectStats(root, nodes, leaves, totalEntries);
        cout << "MTree stats: height=" << height
             << " nodes=" << nodes << " leaves=" << leaves
             << " totalEntries=" << totalEntries << "\n";
    }

private:
    // ------------------- UTIL -------------------

    static float safe_dist(const MTreeObject &a, const MTreeObject &b)
    {
        return mtree_distance(a, b);
    }

    // ---------------- Tipos da fila de prioridade ----------------

    using PQEntry = pair<float, pair<shared_ptr<Node>, size_t>>;

    struct PQCompare
    {
        bool operator()(const PQEntry &a, const PQEntry &b) const
        {
            return a.first > b.first; // min-heap
        }
    };

    using PQ = priority_queue<PQEntry, vector<PQEntry>, PQCompare>;

    // Empilha as entradas do nó na PQ com seus limites inferiores

    void pushNodeEntriesToPQ(const shared_ptr<Node> &node,
                             const MTreeObject &query,
                             PQ &pq)
    {
        for (size_t i = 0; i < node->entries.size(); ++i)
        {
            const Entry &e = node->entries[i];

            float dqp = safe_dist(query, e.routingObj);
            float lb = (e.child ? max(0.0f, dqp - e.radius) : dqp);

            pq.emplace(lb, make_pair(node, i));
        }
    }

    // Altura da árvore (raiz = 1)
    int computeHeight(const shared_ptr<Node> &node)
    {
        if (!node)
            return 0;
        if (node->isLeaf)
            return 1;
        int h = 0;
        for (auto &e : node->entries)
        {
            if (e.child)
            {
                h = max(h, 1 + computeHeight(e.child));
            }
        }
        return h;
    }

    void collectStats(const shared_ptr<Node> &node, size_t &nodes, size_t &leaves, size_t &entries)
    {
        if (!node)
            return;
        nodes++;
        if (node->isLeaf)
            leaves++;
        entries += node->entries.size();
        for (auto &e : node->entries)
            if (e.child)
                collectStats(e.child, nodes, leaves, entries);
    }

    // Procura entrada de folha com objectId retorna nó e índice
    pair<shared_ptr<Node>, int> findLeafEntry(const shared_ptr<Node> &node, int objectId)
    {
        if (!node)
            return {nullptr, -1};
        if (node->isLeaf)
        {
            for (int i = 0; i < (int)node->entries.size(); ++i)
                if (node->entries[i].routingObj.id == objectId)
                    return {node, i};
            return {nullptr, -1};
        }

        for (auto &e : node->entries)
        {
            if (e.child)
            {
                auto res = findLeafEntry(e.child, objectId);
                if (res.first)
                    return res;
            }
        }
        return {nullptr, -1};
    }

    // ------------------- Inserção -------------------

    void insertRecursive(const shared_ptr<Node> &node, const MTreeObject &obj)
    {
        if (node->isLeaf)
        {
            Entry ent;
            ent.routingObj = obj;
            ent.child = nullptr;
            ent.radius = 0.0f;
            node->entries.push_back(ent);

            if (node->entries.size() > M)
            {
                handleOverflow(node);
            }
            else
            {
                updateCoveringRadiiUpwards(node, obj);
            }
            return;
        }

        // Nó interno: escolher subárvore
        size_t bestIdx = chooseSubtree(node, obj);
        assert(bestIdx < node->entries.size());
        auto child = node->entries[bestIdx].child;
        insertRecursive(child, obj); // splits e propagação já tratados recursivamente

        // Atualiza raio de cobertura da entrada escolhida (se ainda apontar para o mesmo filho)
        // O filho pode ter sido dividido e o entry original removido.
        // Em vez de acessar diretamente pelo índice salvo, buscamos o entry que aponta para child.
        for (auto &ent : node->entries)
        {
            if (ent.child == child)
            {
                // debug desativado
                ent.radius = computeCoveringRadius(ent);
                break;
            }
        }
    }

    // Escolhe subarvore minimizando o aumento do raio de cobertura
    size_t chooseSubtree(const shared_ptr<Node> &node, const MTreeObject &obj)
    {
        float bestInc = numeric_limits<float>::infinity();
        float bestDist = numeric_limits<float>::infinity();
        size_t bestIdx = 0;

        for (size_t i = 0; i < node->entries.size(); ++i)
        {
            const Entry &e = node->entries[i];
            float dist = safe_dist(obj, e.routingObj);
            float inc = 0.0f;
            if (dist > e.radius)
                inc = dist - e.radius;
            if (inc < bestInc ||
                (fabs(inc - bestInc) < 1e-6 && (e.radius < node->entries[bestIdx].radius ||
                                                (fabs(e.radius - node->entries[bestIdx].radius) < 1e-6 && dist < bestDist))))
            {
                bestInc = inc;
                bestDist = dist;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // Raio de cobertura para uma entrada
    float computeCoveringRadius(const Entry &e)
    {
        if (!e.child)
            return 0.0f;
        return computeCoveringRadius(e.child, e.routingObj);
    }

    float computeCoveringRadius(const shared_ptr<Node> &node, const MTreeObject &pivot)
    {
        float maxD = 0.0f;
        if (node->isLeaf)
        {
            for (auto &ent : node->entries)
            {
                float d = safe_dist(pivot, ent.routingObj);
                if (d > maxD)
                    maxD = d;
            }
            return maxD;
        }
        else
        {
            for (auto &ent : node->entries)
            {
                // Aproximação pelo pivô do filho + raio do filho (limite superior)
                float d = safe_dist(pivot, ent.routingObj);
                if (d + ent.radius > maxD)
                    maxD = d + ent.radius;
            }
            return maxD;
        }
    }

    float computeCoveringRadius(const shared_ptr<Node> &node)
    {
        // heuristica: pega o primeiro entry como pivô
        if (!node)
            return 0.0f;
        if (node->entries.empty())
            return 0.0f;
        MTreeObject pivot = node->entries[0].routingObj;
        return computeCoveringRadius(node, pivot);
    }

    // Atualiza raios na cadeia de ancestrais para acomodar obj
    void updateCoveringRadiiUpwards(const shared_ptr<Node> &node, const MTreeObject &obj)
    {
        auto cur = node;
        while (auto p = cur->parent.lock())
        {
            // encontra qual entry no pai aponta para cur
            for (auto &ent : p->entries)
            {
                if (ent.child && ent.child == cur)
                {
                    float d = safe_dist(ent.routingObj, obj);
                    if (d > ent.radius)
                        ent.radius = d;
                    break;
                }
            }
            cur = p;
        }
    }

    // ------------------- Overflow / Split -------------------

    void handleOverflow(const shared_ptr<Node> &node)
    {
        // Divide o nó em dois; se for a raiz, cria nova raiz
        auto [n1, n2] = splitNode(node);

        if (node == root)
        {
            // nova raiz
            auto newRoot = make_shared<Node>(false);
            Entry e1, e2;
            e1.routingObj = n1->entries[0].routingObj;
            e1.child = n1;
            e1.radius = computeCoveringRadius(n1, e1.routingObj);
            e2.routingObj = n2->entries[0].routingObj;
            e2.child = n2;
            e2.radius = computeCoveringRadius(n2, e2.routingObj);
            n1->parent = newRoot;
            n2->parent = newRoot;
            newRoot->entries = {e1, e2};
            root = newRoot;
            newRoot->isLeaf = false;
            return;
        }

        // Substitui a entrada antiga do pai por duas novas (n1 e n2)
        auto parent = node->parent.lock();
        assert(parent);
        // encontra índice do node no pai
        int idx = -1;
        for (int i = 0; i < (int)parent->entries.size(); ++i)
            if (parent->entries[i].child == node)
            {
                idx = i;
                break;
            }
        assert(idx != -1);

        // remove a entrada antiga e insere duas novas
        parent->entries.erase(parent->entries.begin() + idx);

        Entry e1, e2;
        e1.routingObj = n1->entries[0].routingObj;
        e1.child = n1;
        e1.radius = computeCoveringRadius(n1, e1.routingObj);
        e2.routingObj = n2->entries[0].routingObj;
        e2.child = n2;
        e2.radius = computeCoveringRadius(n2, e2.routingObj);
        n1->parent = parent;
        n2->parent = parent;

        // insert novas entradas
        parent->entries.push_back(e1);
        parent->entries.push_back(e2);

        // Se o pai estourar, divide recursivamente
        if (parent->entries.size() > M)
            handleOverflow(parent);
    }

    // Split: escolhe par de pivôs e particiona as entradas
    pair<shared_ptr<Node>, shared_ptr<Node>> splitNode(const shared_ptr<Node> &node)
    {
        // Coleta todas as entradas do nó antigo
        vector<Entry> all = node->entries;
        size_t n = all.size();
        assert(n >= 2);
        assert(n == M + 1 || n > M); // overflow esperado

        // Par mais distante
        float maxD = -1.0f;
        size_t a = 0, b = 1;

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                float d = safe_dist(all[i].routingObj, all[j].routingObj);
                if (d > maxD)
                {
                    maxD = d;
                    a = i;
                    b = j;
                }
            }
        }

        // Cria novos nós (mantém tipo folha/interno)
        auto nodeA = make_shared<Node>(node->isLeaf);
        auto nodeB = make_shared<Node>(node->isLeaf);
        NODE_INSTANCES.fetch_add(2, memory_order_relaxed);
        SPLIT_COUNT.fetch_add(1, memory_order_relaxed);
        nodeA->entries.reserve(n); // reserva máxima possível
        nodeB->entries.reserve(n);

        // Copia o ponteiro para o pai
        auto parent = node->parent.lock();
        nodeA->parent = parent;
        nodeB->parent = parent;

        // Seeds promovidos
        Entry seedA = std::move(all[a]);
        Entry seedB = std::move(all[b]);

        // Seeds podem ser de folha ou internas; preserva filhos quando necessário
        if (node->isLeaf)
        {
            seedA.child = nullptr;
            seedB.child = nullptr;
        }

        // Insere seeds
        nodeA->entries.push_back(std::move(seedA));
        nodeB->entries.push_back(std::move(seedB));

        // Marca entradas já usadas
        vector<bool> assigned(n, false);
        assigned[a] = true;
        assigned[b] = true;

        // Distribui demais entradas
        for (size_t i = 0; i < n; ++i)
        {
            if (assigned[i])
                continue;

            float dA = safe_dist(all[i].routingObj, nodeA->entries[0].routingObj);
            float dB = safe_dist(all[i].routingObj, nodeB->entries[0].routingObj);

            if (dA < dB)
                nodeA->entries.push_back(std::move(all[i]));
            else
                nodeB->entries.push_back(std::move(all[i]));

            assigned[i] = true;
        }

        // Garante preenchimento mínimo
        if (nodeA->entries.size() < m)
        {
            moveEntriesToFill(nodeB, nodeA, m - nodeA->entries.size(), nodeA->entries[0].routingObj);
        }
        else if (nodeB->entries.size() < m)
        {
            moveEntriesToFill(nodeA, nodeB, m - nodeB->entries.size(), nodeB->entries[0].routingObj);
        }

        // Atualiza parent nos filhos das entradas internas
        if (!node->isLeaf)
        {
            for (auto &e : nodeA->entries)
                if (e.child)
                    e.child->parent = nodeA;

            for (auto &e : nodeB->entries)
                if (e.child)
                    e.child->parent = nodeB;
        }

        // Invalida o nó antigo
        node->entries.clear();

        // Retorna os dois novos nós
        return {nodeA, nodeB};
    }

    // Move 'count' entradas de src para dst (escolhendo as mais distantes do pivô de dst)
    void moveEntriesToFill(shared_ptr<Node> &src, shared_ptr<Node> &dst, size_t count, const MTreeObject &dstPivot)
    {
        // compute distances of src entries to dstPivot and sort descending
        struct Item
        {
            size_t idx;
            float dist;
        };
        vector<Item> items;
        items.reserve(src->entries.size());
        if (count == 0 || src->entries.empty())
            return;
        if (count > src->entries.size())
            count = src->entries.size(); // proteção
        for (size_t i = 0; i < src->entries.size(); ++i)
        {
            items.push_back({i, safe_dist(src->entries[i].routingObj, dstPivot)});
        }
        sort(items.begin(), items.end(), [](const Item &a, const Item &b)
             { return a.dist > b.dist; });

        size_t moved = 0;
        // move top 'count' items
        for (auto &it : items)
        {
            if (moved >= count)
                break;
            dst->entries.push_back(std::move(src->entries[it.idx]));
            moved++;
        }
        // Remove entradas movidas de src reconstruindo o vetor
        vector<Entry> remaining;
        vector<bool> movedIdx(src->entries.size(), false);
        for (size_t i = 0; i < count && i < items.size(); ++i)
            movedIdx[items[i].idx] = true;
        for (size_t i = 0; i < src->entries.size(); ++i)
            if (!movedIdx[i])
                remaining.push_back(std::move(src->entries[i]));
        src->entries.swap(remaining);

        // Verificação pos-move
    }

    // ------------------- Range Query -------------------

    void rangeRecursive(const shared_ptr<Node> &node, const MTreeObject &query, float radius, vector<MTreeObject> &out)
    {
        if (!node)
            return;
        if (node->isLeaf)
        {
            for (auto &e : node->entries)
            {
                float d = safe_dist(query, e.routingObj);
                if (d <= radius)
                    out.push_back(e.routingObj);
            }
            return;
        }

        for (auto &e : node->entries)
        {
            float dqp = safe_dist(query, e.routingObj);
            if (dqp <= radius + e.radius)
            {
                // Subarvore pode conter resultados
                rangeRecursive(e.child, query, radius, out);
            }
        }
    }

    // ------------------- Underflow / Remoção -------------------

    // Se underflow (entries < m) e não for raiz remove do pai e reinsere
    void handleUnderflow(const shared_ptr<Node> &node)
    {
        if (node == root)
        {
            // se a raiz tem apenas um filho e é interna, torna esse filho a nova raiz
            if (!root->isLeaf && root->entries.size() == 1 && root->entries[0].child)
            {
                root = root->entries[0].child;
                root->parent.reset();
            }
            return;
        }

        if (node->entries.size() >= m)
        {
            // sem underflow
            return;
        }

        // remove no do pai e coleta suas entradas para reinserção
        auto parent = node->parent.lock();
        assert(parent);
        // encontra e remove a entrada do pai que referencia o nó
        int idx = -1;
        for (int i = 0; i < (int)parent->entries.size(); ++i)
            if (parent->entries[i].child == node)
            {
                idx = i;
                break;
            }
        assert(idx != -1);
        // remove o entry
        parent->entries.erase(parent->entries.begin() + idx);

        // coleta todos os objetos folha sob o nó (se interno, coleta descendentes folha)
        vector<MTreeObject> toReinsert;
        collectLeafObjects(node, toReinsert);

        // reinsere-os na árvore começando pela raiz
        for (auto &obj : toReinsert)
            insert(obj);

        // após remoção, talvez o pai também tenha underflow
        handleUnderflow(parent);
    }

    void collectLeafObjects(const shared_ptr<Node> &node, vector<MTreeObject> &out)
    {
        if (!node)
            return;
        if (node->isLeaf)
        {
            for (auto &e : node->entries)
                out.push_back(e.routingObj);
            return;
        }
        for (auto &e : node->entries)
            collectLeafObjects(e.child, out);
    }
};

// Definições inline dos membros estáticos, C++17 inline variable evita múltiplas definições
inline atomic<size_t> MTree::NODE_INSTANCES{0};
inline atomic<size_t> MTree::SPLIT_COUNT{0};

#endif // MTREE_HPP
