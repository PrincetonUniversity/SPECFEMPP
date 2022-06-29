#include <iostream>
#include <cmath>
#include <stdexcept>
#include "../include/gll_library.h"

gll_library::gll::gll()
    : alpha(0.0), beta(0.0), ngll(5){};

gll_library::gll::gll(const double alpha, const double beta, const int ngll)
    : alpha(alpha), beta(beta), ngll(ngll){}

double gll_library::pnleg(const double z, const int n){
    // Generate Lagendre polynomials using recurrance relation
    // (l+1)P_(l+1)(x)-(2l+1)xP_l(x)+lP_(l-1)(x)=0 

    if (n==0) throw std::invalid_argument("value of n > 0");
    
    double p1, p2, p3, double_k;

    p1 = 1.0;
    p2 = z;
    p3 = p2;

    for (int k = 1; k < n; k++){
        double_k = static_cast<double>(k);
        p3  = ((2.0*double_k+1.0)*z*p2 - double_k*p1)/(double_k+1.0);
        p1  = p2;
        p2  = p3;
    }

    return p3;
}

double gll_library::pnglj(const double z, const int n){

    if (n==0) throw std::invalid_argument("value of n > 0");

    double glj_value;
    
    if (std::abs(z+1.0) > 1e-9){
        glj_value = (gll_library::pnleg(z,n)+gll_library::pnleg(z,n+1))/(1.0+z);
    } else {
        glj_value = (static_cast<double>(n)+1.0)*std::pow(-1.0,n);
    }

    return glj_value;
}

double gll_library::pndleg(const double z, const int n){

    if (n==0) throw std::invalid_argument("value of n > 0");

    double p1,p2,p1d,p2d,p3,p3d,double_k;

    p1 = 1.0;
    p2 = z;
    p1d = 0.0;
    p2d = 1.0;
    p3d = 1.0;

    for (int k = 1; k < n; k++){
        double_k = static_cast<double>(k);
        p3 = ((2.0*double_k+1.0)*z*p2 - double_k*p1)/(double_k+1.0);
        p3d = ((2.0*double_k+1.0)*p2 + (2.0*double_k+1.0)*z*p2d 
                - double_k*p1d)/(double_k+1.0);
        p1 = p2;
        p2 = p3;
        p1d = p2d;
        p2d = p3d;
    }

    return p3d;
}

double gll_library::pndglj(const double z, const int n){

    if (n==0) throw std::invalid_argument("value of n > 0");

    double glj_deriv;

    if (std::abs(z+1.0) > 1e-9){
        glj_deriv = (gll_library::pndleg(z,n)+gll_library::pndleg(z,n+1))/(1.0+z) 
                        - (gll_library::pnleg(z,n)+gll_library::pnleg(z,n+1))/((1.0+z)*(1.0+z));
    } else {
        glj_deriv = gll_library::pnleg(-1.0,n)+gll_library::pnleg(-1.0,n+1);
    }

    return glj_deriv;
}