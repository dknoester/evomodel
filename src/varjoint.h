/* evomodel.h
 *
 * This file is part of EvoModel.
 *
 * Copyright 2014 David B. Knoester.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _EVOMODEL_H_
#define _EVOMODEL_H_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cmath>
#include <map>
#include <vector>

#include <ea/meta_data.h>
#include <ea/fitness_function.h>
using namespace ealib;

LIBEA_MD_DECL(EVOMODEL_M, "evomodel.m", int);
LIBEA_MD_DECL(EVOMODEL_N, "evomodel.n", int);
LIBEA_MD_DECL(EVOMODEL_MIN, "evomodel.min", double);
LIBEA_MD_DECL(EVOMODEL_MAX, "evomodel.max", double);
LIBEA_MD_DECL(EVOMODEL_MEAN1, "evomodel.mean1", double);
LIBEA_MD_DECL(EVOMODEL_STD1, "evomodel.std1", double);
LIBEA_MD_DECL(EVOMODEL_MEAN2, "evomodel.mean2", double);
LIBEA_MD_DECL(EVOMODEL_STD2, "evomodel.std2", double);


/* \todo merge this with probability mass function code in math/information.h
 */

//! Type for an empirical distribution function.
typedef std::map<double,double> edf_type;

//! Returns the empirical distribution function for the range [f,l).
template <typename ForwardIterator>
edf_type make_edf(ForwardIterator f, ForwardIterator l) {
    double n=std::distance(f,l);
    edf_type e;
    for( ; f!=l; ++f) {
        e[*f] += 1.0; // map of value -> counter
    }
    double cn=0.0;
    for(edf_type::iterator i=e.begin(); i!=e.end(); ++i) {
        cn += i->second;
        i->second = cn / n; // map of value -> P(sample <= value)
    }
    return e;
}

/*! Returns the probability for event k in EDF e.
 */
double edf_prob(const edf_type::key_type& k, const edf_type& c) {
    edf_type::const_iterator i=c.lower_bound(k); // 1st element whose key is >= k, end if not found
    if(i == c.end()) {
        // everything in c is <= k:
        return 1.0;
    }
    if(i->first == k) {
        // c held an entry for exactly k:
        return i->second;
    }
    if(i == c.begin()) {
        // nothing in c <= k:
        return 0.0;
    }
    
    // if we get here, we know that i->first > k; we want the largest i->first
    // that is not greater than k, so we have to back it up 1:
    --i;
    return i->second;
}


/*! Calculate the two-sample Kolmogorov-Smirnov test statistic for the given sequences.
 
 F_n(x) = empirical distribution function
 
 D_{n,n'} = sup_x | F_{1,n}(x) - F_{2,n'}(x) |
 
 In words, this is the supremum (maximum) of the differences between the empirical
 distribution functions for sequences s1 and s2.
 
 \warning: This statement (from wikipedia) seems a bit off:
 H_0 (that they are the same) is rejected at level \alpha if:
 D_{n,n'} > c(\alpha) \sqrt((n+n')/(nn'))
 
 If doing hypothesis testing with this, beware.
 */
template <typename Sequence1, typename Sequence2>
double kolmogorov_smirnov_test(Sequence1& s1, Sequence2& s2) {
    edf_type e1=make_edf(s1.begin(), s1.end());
    edf_type e2=make_edf(s2.begin(), s2.end());
    double dnn=0.0;
    
    // e1 -> e2
    for(edf_type::iterator i=e1.begin(); i!=e1.end(); ++i) {
        double cp1=i->second;
        double cp2=edf_prob(i->first,e2);
        dnn = std::max(dnn, std::abs(cp1-cp2));
    }
    // e2 -> e1  (have to check both ways, as e1 and e2 may hold different values).
    for(edf_type::iterator i=e2.begin(); i!=e2.end(); ++i) {
        double cp1=edf_prob(i->first,e1);
        double cp2=i->second;
        dnn = std::max(dnn, std::abs(cp1-cp2));
    }
    return dnn;
}

double pnormal(double x, double mu, double sigma) {
    // normal pdf:
    // f(x,mu,sigma) =
    double p = (1.0/(sigma*sqrt(2.0*M_PI)))*exp(-(pow(x-mu,2.0))/(2.0*pow(sigma,2.0)));
    return p;
}

/*! Fitness function that rewards for decreasing D_{n,n'}, the Kolmogorov-Smirnov
 distance, between a target distribution and a distribution generated from parameters
 embedded in a genome.
 */
struct joint_ks : fitness_function<unary_fitness<double>, constantS, stochasticS> {
    typedef boost::numeric::ublas::matrix<double> matrix_type;
    typedef boost::numeric::ublas::matrix_row<matrix_type> row_type;
    
    matrix_type _IC; //!< Matrix of initial conditions.
    
    //! Initialize the fitness function.
    template <typename RNG, typename EA>
    void initialize(RNG& rng, EA& ea) {
        using namespace std;
        double mu1=get<EVOMODEL_MEAN1>(ea), sigma1=get<EVOMODEL_STD1>(ea);
        double mu2=get<EVOMODEL_MEAN2>(ea), sigma2=get<EVOMODEL_STD2>(ea);
        
        typename RNG::real_rng_type r=rng.uniform_real_rng(get<EVOMODEL_MIN>(ea), get<EVOMODEL_MAX>(ea));
        
        _IC.resize(get<EVOMODEL_M>(ea), get<EVOMODEL_N>(ea));
        for(std::size_t i=0; i<_IC.size1(); ++i) {
            for(std::size_t j=0; j<_IC.size2(); ++j) {
                bool reject=true;
                while(reject) {
                    double x = r();
                    double p = rng.p();
                    if((p < pnormal(x,mu1,sigma1)) || (p < pnormal(x,mu2,sigma2))) {
                        reject = false;
                        _IC(i,j) = x;
                    }
                }
            }
        }
    }
    
    //! Calculate fitness.
	template <typename Individual, typename RNG, typename EA>
	double operator()(Individual& ind, RNG& rng, EA& ea) {
        // sample generated by this individual:
        std::vector<double> s(get<EVOMODEL_N>(ea), 0.0);
        
        std::vector<std::pair<double,double> > dist_list;
        typename EA::representation_type& repr=ind.repr();
        for(std::size_t i=0; i<(repr.size()/2); ++i) { // note: might be one left over! (which is ignored)
            dist_list.push_back(std::make_pair(repr[i*2],repr[i*2+1]));
        }
        
        typename RNG::real_rng_type r=rng.uniform_real_rng(get<EVOMODEL_MIN>(ea), get<EVOMODEL_MAX>(ea));
        
        for(std::size_t i=0; i<s.size(); ++i) {
            bool reject=true;
            while(reject) {
                double x = r();
                double p = rng.p();
                for(std::size_t j=0; j<dist_list.size(); ++j) {
                    if(p < pnormal(x,dist_list[j].first,dist_list[j].second)) {
                        reject = false;
                        s[i] = x;
                    }
                }
            }
        }
        
        // sample we're testing it with:
        row_type row(_IC, rng(_IC.size1()));
        
        // fitness is the inverted ks test statistic
        // in words, this is the maximal difference between empirical distribution
        // functions (which we want to minimize):
        return 1.0 / kolmogorov_smirnov_test(s, row);
    }
};


#endif
