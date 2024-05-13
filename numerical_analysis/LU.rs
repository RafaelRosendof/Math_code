// read the equations 
//construct the coefficient matrix A and the rigth-hand side vector b
//perform the LU decomposition on matrix A to obtain matrices L and U
//Store this matrices 
//output the matrices 
use std::io;

fn recebe_sistema() -> Vec<f64>{
    println!("Recebendo os coeficientes, coloque os separados e sem virgula");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Falhou em ler a linha");

    input.trim().split_whitespace()
    .map(|s| s.parse().expect("Falha em passar o input par float"))
    .collect()
}


fn main(){

    println!("Digite quantos sistemas queres: ");
    let mut x = String::new();
    io::stdin().read_line(&mut x).expect("falhou em ler a linha ");
    let num_system: usize =  x.trim().parse().expect("falha em processar o input ");

    let mut system: Vec<(Vec<f64> , Vec<f64>)> = Vec::new();

    for i in 0..num_system{
        println!("Digite o sistema {}, apenas os coeficientes: ", i+1);
        let coefs = recebe_sistema();
        let (A,b) = coefs.split_at(coefs.len() - 1);
        system.push((A.to_vec() , b.to_vec()));

 
    }

    for i in &system{
        println!("   {:?}   ", i);
    }
}