fn func(x: f64) -> f64 {
    x.powf(6.0) - 10.0 * x.powf(5.0) + 35.0 * x.powf(4.0) - 50.0 * x.powf(3.0) + 25.0 * x.powf(2.0) - 5.0 * x
}

fn derivada_fun(x: f64) -> f64 {
    6.0 * x.powf(5.0) - 50.0 * x.powf(4.0) + 140.0 * x.powf(3.0) - 150.0 * x.powf(2.0) + 50.0 * x - 5.0
}

fn newton(mut x: f64 , erro: f64) -> f64 {

    let mut iter = 0;
    //algoritmo do newton 

    let mut h = func(x) / derivada_fun(x);

    while h.abs() > erro{

        h = func(x)/derivada_fun(x);
        println!("Iteração {}  com um erro de {}",iter,h);
        
        //fazendo o x+1
        x = x-h;
        iter +=1;
    }

    println!("O valor da raiz é {}",x);

    return x
}

fn main(){
    //como receber inputs em rust é chato pra kct 
    //vou colocar diretamente aqui 
    let erro = 1e-7;
    let x = 2.0;

    newton(x , erro);

}