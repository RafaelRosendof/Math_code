import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MapReduceStream{

    public static void main(String [] args){

        Function<Double, Double> g = x -> x + 0.1*x; //reduce
        Function<Double, Double> f = x -> x * x; //map

        List<Double> list = new ArrayList<>();

        for(Double i = 0.0; i < 100000; i++){
            list.add(i * 2);
        }

        //muito bom com o parallelStream, fazer a integral com o erro
        //e testar com o java.util.concurrent.ForkJoinPool e RecursiveTaks
        List<Double> novaLista = list.parallelStream().map(f).collect(Collectors.toList());
        Double res = list.parallelStream().reduce(0.0 , (x , y) -> g.apply(x) + g.apply(y));

        System.out.println("Lista original: " + list);
        System.out.println("Lista após o mapeamento: " + novaLista);
        System.out.println("Resultado da redução: " + res);
    }
}