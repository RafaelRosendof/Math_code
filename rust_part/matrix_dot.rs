//array pagina 100
use std::io;

fn Matrix_dot(matA: Vec<Vec<i32>> , matB: Vec<Vec<i32>>) -> Vec<Vec<i32>>{

    let linhas_a = matA.len();
    let cols_a = matA[0].len();
    let cols_b = matB[0].len(); 

    assert_eq!(cols_a , matB[0].len() , "Número de colunas de A tem que ser igual as linhas de B: ");

    let mut resultado = vec![vec![0;cols_b];linhas_a];

    for i in 0..cols_a{
        for ii in 0..cols_b{
            for iii in 0..cols_a{
                resultado[i][ii] += matA[i][iii] * matB[iii][ii];
            } 
        }
    }
    resultado
}

fn main(){
    println!("Fazneod multiplicação de matrizes ");

    println!("Digite quantas colunas da matriz A: ");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("falhou em ler ");

    let s: usize = input.trim().parse().expect("Entrada inválida");

    println!("Digite as quantas linhas da matriz A: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("falhou em ler ");
    let n: usize = input.trim().parse().expect("Entrada inválida");

    println!("Digite as quantas colunas da matriz B: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("falhou em ler ");
    let d: usize = input.trim().parse().expect("Entrada inválida");

    let mut matA = vec![vec![0;s];n];
    let mut matB = vec![vec![0;d];s];

    //fill the matrix A and B
    println!("Digite os elementos de A: ");
    for i in 0..s{
        for j in 0..n{
            println!("Digite o elemento da posição ({} , {}): " , i+1,j+1);
            input.clear();
            io::stdin().read_line(&mut input).expect("Falhou em ler ");
            let elem: i32 = input.trim().parse().expect("Entrada inválida: ");
            matA[j][i] = elem;
        }
    }

    println!("Digite os elementos de B: ");
    for i in 0..d{
        for j in 0..s{
            println!("Digite o elemento da posição ({} , {}): ", j+1,i+1);
            input.clear();
            io::stdin().read_line(&mut input).expect("Falhou em ler ");
            let elem: i32 = input.trim().parse().expect("Entrada inválida ");
            matB[j][i] = elem;
        }
    }

    // Imprimir as matrizes
    println!("Matriz A:");
    for linha in &matA {
        for &elemento in linha {
            print!("{} ", elemento);
        }
        println!();
    }

    //verificando a matriz B
    println!("MATRIZ B");
    for lin in &matB{
        for &elem in lin{
            print!("{} ",elem);
        }
        println!();
    }

    println!("Fazendo a multiplicação agora \n\n\n");


    let resultado = Matrix_dot(matA , matB);

    println!("Resultado:");
    for row in &resultado {
        for &elem in row {
            print!("{} ", elem);
        }
        println!();
    }

}