import java.util.function.Function;
import java.util.Scanner;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.stream.IntStream;
import java.math.MathContext;

public class IntegralStream{

    public static void main(String[] args){

        Scanner let = new Scanner(System.in);

        System.out.println("Integral com Streams paralelas");

        Function<BigDecimal , BigDecimal> f = x ->  BigDecimal.valueOf(Math.log10(Math.sqrt(x.doubleValue() * x.doubleValue() + x.doubleValue()) + 1));

        System.out.println("Digite o valor de a: ");
        BigDecimal a = let.nextBigDecimal();

        System.out.println("Digite o valor de b: ");
        BigDecimal b = let.nextBigDecimal();

        System.out.println("Digite o valor de n: ");
        int n = let.nextInt();

        System.out.println("Digite a tolerancia: ");
        BigDecimal tol = let.nextBigDecimal();

        //metodo do trapézio considerando a diferença entre dois trapezios sucessivos
        // sendo o criterio de parada o erro
        //fazer com o parallelStream

        long inicio = System.nanoTime();
        MathContext mc = new MathContext(30 , RoundingMode.HALF_EVEN);

        BigDecimal integral = trapezios(f , a , b , n , tol , mc);
        
        long fim = System.nanoTime();

        long durou = (fim - inicio) / 1_000_000;

        let.close();
        System.out.println("O valor da INTEGRAL É : " + integral);

        System.out.println("Tempo de duração: " + durou + "ms");
    }

    public static BigDecimal trapezios(Function<BigDecimal , BigDecimal> f , BigDecimal a , BigDecimal b , int n , BigDecimal tol , MathContext mc){
        BigDecimal previo = BigDecimal.ZERO;
        BigDecimal atual = calculo(f , a , b , n , mc);

        while(atual.subtract(previo).abs().compareTo(tol) > 0){
            n = n * 2;
            previo = atual;
            atual = calculo(f , a , b , n , mc);
        }
        return atual;
    }

    public static BigDecimal calculo(Function<BigDecimal , BigDecimal> f , BigDecimal a , BigDecimal b , int n , MathContext mc){
        BigDecimal h = b.subtract(a).divide(BigDecimal.valueOf(n) , mc);

        BigDecimal sum = IntStream.range(1 , n).parallel().mapToObj(i -> f.apply(a.add(BigDecimal.valueOf(i).multiply(h , mc))))
        .reduce(BigDecimal.ZERO , BigDecimal::add);


        sum = sum.add(f.apply(a).add(f.apply(b)).divide(BigDecimal.valueOf(2) , mc));
        return sum.multiply(h , mc);
    }

}