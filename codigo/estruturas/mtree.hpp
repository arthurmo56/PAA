#pragma once
#include "../descritor/descritor.hpp"
#include <vector>
#include <memory>

struct Point {
    float x, y; // muH, muS
    Record record;
};

struct Rect {
    float x, y;     // centro
    float halfW, halfH; // metade da largura e altura
    bool contains(const Point& p) const {
        return (p.x >= x - halfW && p.x <= x + halfW &&
                p.y >= y - halfH && p.y <= y + halfH);
    }
    bool intersects(const Rect& range) const {
        return !(range.x - range.halfW > x + halfW ||
                 range.x + range.halfW < x - halfW ||
                 range.y - range.halfH > y + halfH ||
                 range.y + range.halfH < y - halfH);
    }
};

class Quadtree {
private:
    Rect boundary;
    int capacity;
    std::vector<Point> points;
    bool divided;

    std::unique_ptr<Quadtree> northeast;
    std::unique_ptr<Quadtree> northwest;
    std::unique_ptr<Quadtree> southeast;
    std::unique_ptr<Quadtree> southwest;

public:
    Quadtree(Rect boundary, int capacity);

    bool insert(const Point& p);
    void subdivide();

    void queryRange(const Rect& range, std::vector<Point>& found) const;
};

Quadtree::Quadtree(Rect boundary, int capacity)
    : boundary(boundary), capacity(capacity), divided(false) {}

bool Quadtree::insert(const Point& p) {
    if (!boundary.contains(p)) return false;

    if (points.size() < capacity) {
        points.push_back(p);
        return true;
    } else {
        if (!divided) subdivide();

        if (northeast->insert(p)) return true;
        if (northwest->insert(p)) return true;
        if (southeast->insert(p)) return true;
        if (southwest->insert(p)) return true;
    }
    return false;
}

void Quadtree::subdivide() {
    float x = boundary.x;
    float y = boundary.y;
    float hw = boundary.halfW / 2.0f;
    float hh = boundary.halfH / 2.0f;

    northeast = std::make_unique<Quadtree>(Rect{x+hw, y-hh, hw, hh}, capacity);
    northwest = std::make_unique<Quadtree>(Rect{x-hw, y-hh, hw, hh}, capacity);
    southeast = std::make_unique<Quadtree>(Rect{x+hw, y+hh, hw, hh}, capacity);
    southwest = std::make_unique<Quadtree>(Rect{x-hw, y+hh, hw, hh}, capacity);

    divided = true;
}

void Quadtree::queryRange(const Rect& range, std::vector<Point>& found) const {
    if (!boundary.intersects(range)) return;

    for (const auto& p : points) {
        if (range.contains(p)) {
            found.push_back(p);
        }
    }

    if (divided) {
        northeast->queryRange(range, found);
        northwest->queryRange(range, found);
        southeast->queryRange(range, found);
        southwest->queryRange(range, found);
    }
}