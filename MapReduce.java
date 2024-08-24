import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.Function;

public class MapReduce {

    private static final int THRESHOLD = 100;
//todo, revisar o código 
    public static void main(String[] args) {
    
        Function<Double, Double> f = x -> x * 0.1*x;
        Function<Double, Double> g = x -> x + x;
        List<Double> list = new ArrayList<>();

        for (Double i = 0.0; i < 10000000; i++) {
            list.add(i * 2);
        }

        ForkJoinPool pool = new ForkJoinPool();
        List<Double> novaLista = pool.invoke(new MapTask(list, f, 0, list.size()));

        Double resultado = pool.invoke(new ReduceTask(novaLista, g, 0, novaLista.size()));

        // Exibindo o resultado
        System.out.println("Lista original: " + list);
        System.out.println("Lista após o mapeamento: " + novaLista);
        System.out.println("Resultado da redução: " + resultado);
    }

    // Task para mapear os valores
    private static class MapTask extends RecursiveTask<List<Double>> {
        private final List<Double> list;
        private final Function<Double, Double> function;
        private final int start;
        private final int end;

        MapTask(List<Double> list, Function<Double, Double> function, int start, int end) {
            this.list = list;
            this.function = function;
            this.start = start;
            this.end = end;
        }

        @Override
        protected List<Double> compute() {
            if (end - start <= THRESHOLD) {
                List<Double> result = new ArrayList<>();
                for (int i = start; i < end; i++) {
                    result.add(function.apply(list.get(i)));
                }
                return result;
            } else {
                int mid = (start + end) / 2;
                MapTask leftTask = new MapTask(list, function, start, mid);
                MapTask rightTask = new MapTask(list, function, mid, end);
                leftTask.fork();
                List<Double> rightResult = rightTask.compute();
                List<Double> leftResult = leftTask.join();
                leftResult.addAll(rightResult);
                return leftResult;
            }
        }
    }

    // Task para reduzir os valores
    private static class ReduceTask extends RecursiveTask<Double> {
        private final List<Double> list;
        private final Function<Double , Double> function;
        private final int start;
        private final int end;

        ReduceTask(List<Double> list, Function<Double, Double> function, int start, int end) {
            this.list = list;
            this.function = function;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Double compute() {
            if (end - start <= THRESHOLD) {
                Double result = 0.0;
                for (int i = start; i < end; i++) {
                    result += function.apply(list.get(i));
                }
                return result;
            } else {
                int mid = (start + end) / 2;
                ReduceTask leftTask = new ReduceTask(list, function, start, mid);
                ReduceTask rightTask = new ReduceTask(list, function, mid, end);
                leftTask.fork();
                Double rightResult = rightTask.compute();
                Double leftResult = leftTask.join();
                return leftResult + rightResult;
            }
        }
    }
}
