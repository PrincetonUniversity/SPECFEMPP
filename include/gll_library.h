
namespace gll_library {
    class gll{
        public:
            gll();
            gll(double alpha, double beta, int ngll);
        
        private:
            float alpha, beta;
            int ngll;
            float pnleg(const float z, const int n);
            float pnglj(const float z, const int n);
            float pndleg(const float z, const int n);
            float pndglj(const float z, const int n);
    };
}