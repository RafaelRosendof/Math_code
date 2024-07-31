import java.util.concurrent.RecursiveTask;
import java.util.function.Function;
import java.util.concurrent.ForkJoinPool;

public class TrapezoidalRule {

    private static final long THRESHOLD = 2100000000;
                                          

    private static final Function<Double , Double> f = x -> Math.log10(Math.sqrt(x*x+x)+1);

    private static class IntegralTask extends RecursiveTask<Double> {
        private final double a;
        private final double b;
        private final int n;

        IntegralTask(double a, double b, int n) {
            this.a = a;
            this.b = b;
            this.n = n;
        }

        @Override
        protected Double compute() {
            if (n <= THRESHOLD) {
                return computeDirectly();
            } else {
                double mid = (a + b) / 2;
                IntegralTask left = new IntegralTask(a, mid, n / 2);
                IntegralTask right = new IntegralTask(mid, b, n / 2);
                left.fork();
                double rightResult = right.compute();
                double leftResult = left.join();
                return leftResult + rightResult;
            }
        }

        private Double computeDirectly() {
            double h = (b - a) / n;
            double integral = (f.apply(a) + f.apply(b)) / 2.0;

            for (int i = 1; i < n; i++) {
                integral += f.apply(a + i * h);
            }

            integral *= h;
            return integral;
        }
    }

    public static double trapezoidalRuleParallel(double a, double b, int n) {
        ForkJoinPool pool = new ForkJoinPool();
        return pool.invoke(new IntegralTask(a, b, n));
    }

    public static void main(String[] args) {
        double a = 2;
        double b = 100;
        int n = 1000000; // Aumentar o número de subintervalos para maior precisão

        double integral = trapezoidalRuleParallel(a, b, n);

        System.out.printf("Integral (ForkJoinPool Trapézio): %.30f%n", integral);
    }
}
