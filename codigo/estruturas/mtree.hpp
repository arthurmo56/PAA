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

using namespace std;

/*
  Implementação de M-Tree:
  - NodeEntry: representa uma entrada em um nó (pode apontar para child ou ser folha com objeto)
  - Node: tem várias entradas
  - Inserção: chooseSubtree (min expand), inserir em folha, split quando overflow (promote por farthest-pair)
  - Range query e k-NN (best-first)
  - Remoção: remove e reinserção das entradas órfãs (reinsert strategy)
*/

class MTree
{
public:
    struct Node;
    struct Entry
    {
        // routing object (pivot) stored here
        MTreeObject routingObj;

        // se apontar para filho
        shared_ptr<Node> child = nullptr;

        // covering radius: máximo d(routingObj, any object in child subtree)
        float radius = 0.0f;

        // distância do routingObj ao pivot do nó pai (útil para persistência). Aqui usamos 0 se root.
        float distToParent = 0.0f;

        // se child == nullptr então esta entry representa um objeto folha,
        // e routingObj é o próprio objeto armazenado na folha.
        bool isLeafEntry() const { return child == nullptr; }
    };

    struct Node
    {
        bool isLeaf = true;
        vector<Entry> entries;
        weak_ptr<Node> parent;

        Node(bool leaf = true) : isLeaf(leaf) {}
    };

    // Configuração
    size_t M = 50; // capacidade máxima (ajuste conforme trabalho)
    size_t m = 25; // mínimo (normalmente M/2)

    shared_ptr<Node> root;

    MTree(size_t maxEntries = 50)
    {
        M = maxEntries;
        m = max((size_t)2, maxEntries / 2);
        root = make_shared<Node>(true);
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

        // push root children
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

    // Remoção (remove objeto exatamente com same id)
    bool remove(int objectId)
    {
        // find leaf entry containing id
        auto [node, idx] = findLeafEntry(root, objectId);
        if (!node)
            return false;
        // remove entry
        MTreeObject orphan = node->entries[idx].routingObj;
        node->entries.erase(node->entries.begin() + idx);

        // If underflow, collect orphaned entries and reinsert
        handleUnderflow(node);
        return true;
    }

    // Debug: print structure (counts)
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

    // ---------------- PQ TYPES --------------------------------------------------

    using PQEntry = pair<float, pair<shared_ptr<Node>, size_t>>;

    struct PQCompare
    {
        bool operator()(const PQEntry &a, const PQEntry &b) const
        {
            return a.first > b.first; // min-heap
        }
    };

    using PQ = priority_queue<PQEntry, vector<PQEntry>, PQCompare>;

    // --------------- pushNodeEntriesToPQ ----------------------

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

    // compute height (root = 1)
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

    // find leaf entry with objectId. returns node shared_ptr and index in node->entries
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
        // internal: traverse all children (could be optimized with bounds)
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

    // ------------------- INSERT -------------------

    void insertRecursive(const shared_ptr<Node> &node, const MTreeObject &obj)
    {
        if (node->isLeaf)
        {
            // create leaf entry
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
                // update ancestors' covering radii
                updateCoveringRadiiUpwards(node, obj);
            }
            return;
        }

        // internal node: choose subtree
        size_t bestIdx = chooseSubtree(node, obj);
        assert(bestIdx < node->entries.size());
        auto child = node->entries[bestIdx].child;
        insertRecursive(child, obj);

        // after insertion, maybe child split changed structure; handle if child's size > M
        if (child->entries.size() > M)
            handleOverflow(child);

        // ensure covering radius of the chosen entry updated
        node->entries[bestIdx].radius = computeCoveringRadius(node->entries[bestIdx]);
    }

    // choose subtree entry index minimizing increase in covering radius
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
            // pick minimal increase, tie-break by smaller radius then smaller dist
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

    // compute covering radius for an entry (max distance between routingObj and all objects in child subtree)
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
                // distance from pivot to child's routingObj plus child's radius may give upper bound,
                // but here we compute exact by exploring downwards for correctness.
                float d = safe_dist(pivot, ent.routingObj);
                if (d + ent.radius > maxD)
                    maxD = d + ent.radius;
            }
            return maxD;
        }
    }

    float computeCoveringRadius(const shared_ptr<Node> &node)
    {
        // heuristic: take first entry as pivot
        if (!node)
            return 0.0f;
        if (node->entries.empty())
            return 0.0f;
        MTreeObject pivot = node->entries[0].routingObj;
        return computeCoveringRadius(node, pivot);
    }

    // update covering radii up the parent chain to accommodate object obj
    void updateCoveringRadiiUpwards(const shared_ptr<Node> &node, const MTreeObject &obj)
    {
        auto cur = node;
        while (auto p = cur->parent.lock())
        {
            // find which entry in parent points to cur
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

    // ------------------- OVERFLOW / SPLIT -------------------

    void handleOverflow(const shared_ptr<Node> &node)
    {
        // split node into two nodes; if node is root, create new root
        auto [n1, n2] = splitNode(node);

        if (node == root)
        {
            // new root
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

        // replace node in its parent by two entries pointing to n1 and n2
        auto parent = node->parent.lock();
        assert(parent);
        // find index of node in parent
        int idx = -1;
        for (int i = 0; i < (int)parent->entries.size(); ++i)
            if (parent->entries[i].child == node)
            {
                idx = i;
                break;
            }
        assert(idx != -1);

        // remove the old entry and insert two new
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

        // insert and possibly adjust order
        parent->entries.push_back(e1);
        parent->entries.push_back(e2);

        // if parent now overflow, handle recursively
        if (parent->entries.size() > M)
            handleOverflow(parent);
    }

    // split node: choose two promoted routing objects and partition entries
    pair<shared_ptr<Node>, shared_ptr<Node>> splitNode(const shared_ptr<Node> &node)
    {
        // Coleta todas as entradas do nó antigo
        vector<Entry> all = node->entries;
        size_t n = all.size();
        assert(n >= 2);

        // (1) Escolhe par mais distante (farthest-pair)
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

        // (2) Cria os novos nós (preservando leafness)
        auto nodeA = make_shared<Node>(node->isLeaf);
        auto nodeB = make_shared<Node>(node->isLeaf);

        // Copia o parent para os novos nós
        auto parent = node->parent.lock();
        nodeA->parent = parent;
        nodeB->parent = parent;

        // (3) Seeds promovidos
        Entry seedA = all[a];
        Entry seedB = all[b];

        // Seeds podem ser entradas de folha ou não.
        // Se não for folha, preserva filho corretamente.
        if (node->isLeaf)
        {
            seedA.child = nullptr;
            seedB.child = nullptr;
        }

        // Inserir seeds como primeiras entradas
        nodeA->entries.push_back(seedA);
        nodeB->entries.push_back(seedB);

        // (4) Marcar entradas já usadas
        vector<bool> assigned(n, false);
        assigned[a] = true;
        assigned[b] = true;

        // (5) Distribuição das outras entradas
        for (size_t i = 0; i < n; ++i)
        {
            if (assigned[i])
                continue;

            float dA = safe_dist(all[i].routingObj, seedA.routingObj);
            float dB = safe_dist(all[i].routingObj, seedB.routingObj);

            if (dA < dB)
                nodeA->entries.push_back(all[i]);
            else
                nodeB->entries.push_back(all[i]);

            assigned[i] = true;
        }

        // (6) Ensure min-fill (corrige desbalanceamento)
        if (nodeA->entries.size() < m)
        {
            moveEntriesToFill(nodeB, nodeA, m - nodeA->entries.size(), seedA.routingObj);
        }
        else if (nodeB->entries.size() < m)
        {
            moveEntriesToFill(nodeA, nodeB, m - nodeB->entries.size(), seedB.routingObj);
        }

        // (7) Se o nó não for folha, atualiza parents dos filhos
        if (!node->isLeaf)
        {
            for (auto &e : nodeA->entries)
                if (e.child)
                    e.child->parent = nodeA;

            for (auto &e : nodeB->entries)
                if (e.child)
                    e.child->parent = nodeB;
        }

        // (8) Invalida o nó antigo para evitar lixo/corrupção
        node->entries.clear();

        // Retorna os dois novos nós
        return {nodeA, nodeB};
    }

    // move count entries from src to dst to ensure dst has required number; choose entries maximizing distance from dstPivot
    void moveEntriesToFill(shared_ptr<Node> &src, shared_ptr<Node> &dst, size_t count, const MTreeObject &dstPivot)
    {
        // compute distances of src entries to dstPivot and sort descending
        struct Item
        {
            size_t idx;
            float dist;
        };
        vector<Item> items;
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
            dst->entries.push_back(src->entries[it.idx]);
            moved++;
        }
        // remove moved entries from src (by rebuilding vector except moved indices)
        vector<Entry> remaining;
        vector<bool> movedIdx(src->entries.size(), false);
        for (size_t i = 0; i < count && i < items.size(); ++i)
            movedIdx[items[i].idx] = true;
        for (size_t i = 0; i < src->entries.size(); ++i)
            if (!movedIdx[i])
                remaining.push_back(src->entries[i]);
        src->entries.swap(remaining);
    }

    // ------------------- RANGE -------------------

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
                // sub-tree may contain results
                rangeRecursive(e.child, query, radius, out);
            }
        }
    }

    // ------------------- UNDERFLOW / REMOVE -------------------

    // if node underflows (entries < m) and not root, remove node from parent and reinsert its entries into tree
    void handleUnderflow(const shared_ptr<Node> &node)
    {
        if (node == root)
        {
            // if root has only one child and is internal, make that child new root (height shrink)
            if (!root->isLeaf && root->entries.size() == 1 && root->entries[0].child)
            {
                root = root->entries[0].child;
                root->parent.reset();
            }
            return;
        }

        if (node->entries.size() >= m)
        {
            // no underflow
            return;
        }

        // remove node from parent and collect its entries for reinsertion
        auto parent = node->parent.lock();
        assert(parent);
        // find and erase parent's entry referencing node
        int idx = -1;
        for (int i = 0; i < (int)parent->entries.size(); ++i)
            if (parent->entries[i].child == node)
            {
                idx = i;
                break;
            }
        assert(idx != -1);
        // remove the entry
        parent->entries.erase(parent->entries.begin() + idx);

        // collect all leaf objects under node (if internal, collect leaf descendants)
        vector<MTreeObject> toReinsert;
        collectLeafObjects(node, toReinsert);

        // reinsert them into tree starting at root
        for (auto &obj : toReinsert)
            insert(obj);

        // after removal maybe parent underflows too
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

#endif // MTREE_HPP
