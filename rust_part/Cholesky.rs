use std::io;

fn Cholesky(matA: Vec<Vec<f64>>)-> Vec<Vec<f64>>{
    let n = matA.len();
    let mut baixo = vec![vec![0.0;n];n];

    for i in 0..n{
        for j in 0..=i{
            let mut sum = 0.0;

            if j == i{
                for k in 0..j{
                    sum += baixo[j][k] * baixo[j][k];
                }
                baixo[j][j] = (matA[j][j] as f64 - sum).sqrt();
            }else{
                for k in 0..j{
                    sum += baixo[i][k] * baixo[j][k];
                }
                baixo[i][j] = (matA[i][j] as f64 - sum)/baixo[j][j];
            }
        }
    }
    baixo

}

fn main(){
    println!("Fazendo a Decomposição de Cholesky");

  //  println!("Digite quantas colunas da matriz A: ");

    let mut input = String::new();
   // io::stdin().read_line(&mut input).expect("Falhou em ler ");

  //  let s: usize = input.trim().parse().expect("Entrada inválida ");

    println!("Digite quantas linhas da Matriz A: ");
  //  input.clear();
    io::stdin().read_line(&mut input).expect("falhou em ler ");
    let n: usize = input.trim().parse().expect("Entrada inválida");

    let mut matA = vec![vec![0.0;n];n];

    println!("Digite os elementos da matriz A: ");

    for i in 0..n{
        for j in 0..n{
            println!("Digite o elemento na posição ({} , {} )", i+1 , j+1);
            input.clear();
            io::stdin().read_line(&mut input).expect("falhou");
           // let n: f64 = input.trim().parse().expect("Inválido");
            matA[i][j] = input.trim().parse().expect("Inválido ");
        }
    }

    println!("\n\nAgora fazendo a decomposição de Cholesky \n");

    let baixo = Cholesky(matA);

    println!("A matriz resultante é \n");
    for i in baixo{
        for val in i{
            print!(" {} ",val);
            
        }
        println!();
    }
}