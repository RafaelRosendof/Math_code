#ifndef HEADER_HPP
#define HEADER_HPP

#include <iostream>
#include <vector>
#include <random>

class GridWorld {
private:
    int rows, cols;  // Dimensões da grade
    int agent_x, agent_y;  // Posição do agente
    std::vector<std::vector<int>> grid;  // Representação da grade
    std::vector<std::pair<int, int>> goal_states;  // Estados terminais
    std::vector<std::pair<int, int>> bad_states;   // Estados de penalidade
    double epsilon; // Parâmetro de exploração para aprendizado

public:
    // Construtor
    GridWorld(int n, int m);

    //~GridWorld();

    
    std::vector<std::vector<int>> create_grid(int n, int m);
    void print_grid() const;

    void print_action();
    
    
    void att_grid();
    
    
    void advance();
    void back();
    void left();
    void right();

    
    double rewardGood();
    double rewardBad();
    double rewardSame();

    
    int choose_action();

    
    void reset();

    void set_epsilon(double e);
    
};

#endif // HEADER_HPP
