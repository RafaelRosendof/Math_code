import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.Function;

public class MapReduce {

    private static final int THRESHOLD = 10;
//todo, revisar o código 
    public static void main(String[] args) {
        Function<Long, Long> f = x -> x * x;
        Function<Long, Long> g = x -> x + x;
        List<Long> list = new ArrayList<>();

        for (long i = 0; i < 70; i++) {
            list.add(i * 2);
        }

        ForkJoinPool pool = new ForkJoinPool();
        List<Long> novaLista = pool.invoke(new MapTask(list, f, 0, list.size()));

        long resultado = pool.invoke(new ReduceTask(novaLista, g, 0, novaLista.size()));

        // Exibindo o resultado
        System.out.println("Lista original: " + list);
        System.out.println("Lista após o mapeamento: " + novaLista);
        System.out.println("Resultado da redução: " + resultado);
    }

    // Task para mapear os valores
    private static class MapTask extends RecursiveTask<List<Long>> {
        private final List<Long> list;
        private final Function<Long, Long> function;
        private final int start;
        private final int end;

        MapTask(List<Long> list, Function<Long, Long> function, int start, int end) {
            this.list = list;
            this.function = function;
            this.start = start;
            this.end = end;
        }

        @Override
        protected List<Long> compute() {
            if (end - start <= THRESHOLD) {
                List<Long> result = new ArrayList<>();
                for (int i = start; i < end; i++) {
                    result.add(function.apply(list.get(i)));
                }
                return result;
            } else {
                int mid = (start + end) / 2;
                MapTask leftTask = new MapTask(list, function, start, mid);
                MapTask rightTask = new MapTask(list, function, mid, end);
                leftTask.fork();
                List<Long> rightResult = rightTask.compute();
                List<Long> leftResult = leftTask.join();
                leftResult.addAll(rightResult);
                return leftResult;
            }
        }
    }

    // Task para reduzir os valores
    private static class ReduceTask extends RecursiveTask<Long> {
        private final List<Long> list;
        private final Function<Long, Long> function;
        private final int start;
        private final int end;

        ReduceTask(List<Long> list, Function<Long, Long> function, int start, int end) {
            this.list = list;
            this.function = function;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Long compute() {
            if (end - start <= THRESHOLD) {
                long result = 0;
                for (int i = start; i < end; i++) {
                    result += function.apply(list.get(i));
                }
                return result;
            } else {
                int mid = (start + end) / 2;
                ReduceTask leftTask = new ReduceTask(list, function, start, mid);
                ReduceTask rightTask = new ReduceTask(list, function, mid, end);
                leftTask.fork();
                long rightResult = rightTask.compute();
                long leftResult = leftTask.join();
                return leftResult + rightResult;
            }
        }
    }
}
