#include <iostream>
#include <cmath>
#include <gll_library.h>

using namespace gll_library;

gll::gll()
    : alpha(0.0), beta(0.0), ngll(5){};

gll::gll(const double alpha, const double beta, const int ngll)
    : alpha(alpha), beta(beta), ngll(ngll){}

float gll::pnleg(const float z, const int n){
    float p1, p2, p3, double_k;

    p1 = 1.0;
    p2 = z;
    p3 = p2;

    for (int k = 1; k < n-1; k++){
        double_k = static_cast<float>(k);
        p3  = ((2.0*double_k+1.0)*z*p2 - double_k*p1)/(double_k+1.0);
        p1  = p2;
        p2  = p3;
    }

    return p3;
}

float gll::pnglj(const float z, const int n){

    float glj_value;
    
    if (std::abs(z+1.0) > 1e-9){
        glj_value = (pnleg(z,n)+pnleg(z,n+1))/(1.0+z);
    } else {
        glj_value = (static_cast<float>(n)+1.0)*std::pow(-1.0,n);
    }

    return glj_value;
}

float gll::pndleg(const float z, const int n){

    float p1,p2,p1d,p2d,p3,p3d,double_k;

    p1 = 1.0;
    p2 = z;
    p1d = 0.0;
    p2d = 1.0;
    p3d = 1.0;

    for (int k = 1; k < n-1; k++){
        double_k = static_cast<float>(k);
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

float gll::pndglj(const float z, const int n){

    float glj_deriv;

    if (std::abs(z+1.0) > 1e-9){
        glj_deriv = (pndleg(z,n)+pndleg(z,n+1))/(1.0+z) 
                        - (pnleg(z,n)+pnleg(z,n+1))/((1.0+z)*(1.0+z));
    } else {
        glj_deriv = pnleg(-1.0,n)+pnleg(-1.0,n+1);
    }

    return glj_deriv;
}