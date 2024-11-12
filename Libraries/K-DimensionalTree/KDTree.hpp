#pragma once

/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 *  https://rosettacode.org/wiki/K-d_tree
 * It is a reimplementation of the C code using C++.
 * It also includes a few more queries than the original
 *
 */

#include <algorithm>
#include <memory>
#include <vector>

#define _D_Dragonian_Lib_KD_Tree_Header namespace DragonianLib { namespace KDTree {
#define _D_Dragonian_Lib_KD_Tree_End } }

_D_Dragonian_Lib_KD_Tree_Header

using point_t = std::vector<float>;
using indexArr = std::vector< size_t >;
using pointIndex = std::pair< std::vector< float >, size_t >;

class KDNode {
public:
    using KDNodePtr = std::shared_ptr< KDNode >;
    size_t index;
    point_t x;
    KDNodePtr left;
    KDNodePtr right;

    // initializer
    KDNode();
    KDNode(const point_t&, const size_t&, const KDNodePtr&,
        const KDNodePtr&);
    KDNode(const pointIndex&, const KDNodePtr&, const KDNodePtr&);
    ~KDNode();

	KDNode(const KDNode&) = default;
	KDNode(KDNode&&) = default;
	KDNode& operator=(const KDNode&) = default;
	KDNode& operator=(KDNode&&) = default;

    // getter
    float coord(const size_t&) const;

    // conversions
    explicit operator bool() const;
    explicit operator point_t();
    explicit operator size_t() const;
    explicit operator pointIndex();
};

using KDNodePtr = std::shared_ptr< KDNode >;

KDNodePtr NewKDNodePtr();

// square euclidean distance
inline float dist2(const point_t&, const point_t&);
inline float dist2(const KDNodePtr&, const KDNodePtr&);

// euclidean distance
inline float dist(const point_t&, const point_t&);
inline float dist(const KDNodePtr&, const KDNodePtr&);

// Need for sorting
class comparer {
public:
    size_t idx;
    explicit comparer(size_t idx_);
    inline bool compare_idx(
        const std::pair< std::vector< float >, size_t >&,  //
        const std::pair< std::vector< float >, size_t >&   //
    ) const;
};

using pointIndexArr = std::vector< pointIndex >;

inline void sort_on_idx(const pointIndexArr::iterator&,  //
    const pointIndexArr::iterator&,  //
    size_t idx);

using pointVec = std::vector<point_t>;

class KDTree {
    KDNodePtr root;
    KDNodePtr leaf;

    KDNodePtr make_tree(const pointIndexArr::iterator& begin,  //
        const pointIndexArr::iterator& end,    //
        const size_t& length,                  //
        const size_t& level                    //
    );

public:
    KDTree() = default;
    explicit KDTree(pointVec point_array);

private:
    static KDNodePtr nearest_(           //
        const KDNodePtr& branch,  //
        const point_t& pt,        //
        const size_t& level,      //
        const KDNodePtr& best,    //
        const float& best_dist   //
    );

    // default caller
    KDNodePtr nearest_(const point_t& pt) const;

public:
    point_t nearest_point(const point_t& pt) const;
    size_t nearest_index(const point_t& pt) const;
    pointIndex nearest_pointIndex(const point_t& pt) const;

private:
    static pointIndexArr neighborhood_(  //
        const KDNodePtr& branch,  //
        const point_t& pt,        //
        const float& rad,        //
        const size_t& level       //
    );

public:
    pointIndexArr neighborhood(  //
        const point_t& pt,       //
        const float& rad) const;

    pointVec neighborhood_points(  //
        const point_t& pt,         //
        const float& rad) const;

    indexArr neighborhood_indices(  //
        const point_t& pt,          //
        const float& rad) const;
};

_D_Dragonian_Lib_KD_Tree_End