#include <iostream>
#include <cmath>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace mp = boost::multiprecision;

class Secante {

public:
    //construtor da classe 
    Secante(double x0 = 0.0, double x1 = 0.0, mp::cpp_dec_float_50 tolerance = 0.0000001)
        : x(x0), x_1(x1), erro(tolerance) {}  // inicialização dinâmica em c++ do construtor

    // métodos get e sets
    double getX() const { return this->x; } //Esse this -> é pq passa a referência como pointer 
    void setX(double new_x) { this->x = new_x; }

    double getX1() const { return this->x_1; }
    void setX1(double new_x1) { this->x_1 = new_x1; }

    mp::cpp_dec_float_50 getErro() const { return this->erro; }
    void setErro(mp::cpp_dec_float_50 new_erro) { this->erro = new_erro; }

    // função para fazer a iteração
    double fun(double x) {
        return (x*x*x - 5*x + 1); //Aqui vc coloca o que quiser, eu vou tentar com essa aqui x³ - 5x + 1 = 0
    }

    double iterativo() {
        double x_new;
        int iteration = 0;

        while (std::abs(fun(x) - fun(x_1)) > erro) {
            x_new = x_1 - ((fun(x_1) * (x_1 - x)) / (fun(x_1) - fun(x)));
            x = x_1;
            x_1 = x_new;
            iteration++;

            std::cout << "Iteration: " << iteration << ", Root: " << x_new << std::endl;
        }

        return x_new;
    }

private:
    double x;
    double x_1;
    mp::cpp_dec_float_50 erro; 
};

int main() {

    
    double a , b;
    std::cout<<"Digite o intervalo de busca: ";
    std::cin>> a >> b;

  //aqui podia ser usado os métodos get e sets para inicializar as varáveis, mas to com preguiça

  //aqui tbm poderia, nem sei pq escrevi, agora já era kkkkkkkkk
    mp::cpp_dec_float_50 erro;
    std::cout<<"Digite o erro da entrada: ";
    std::cin>> erro;

    Secante sec(a, b, erro);
    double root = sec.iterativo();

    std::cout << "Approximate : " << root << std::endl;

    return 0;
}
