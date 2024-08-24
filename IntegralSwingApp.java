import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.function.Function;
import java.util.stream.IntStream;

public class IntegralSwingApp {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(IntegralSwingApp::createAndShowGUI);
    }

    private static void createAndShowGUI() {
        JFrame frame = new JFrame("Cálculo de Integral com Swing");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 300);

        JPanel panel = new JPanel();
        frame.add(panel);
        placeComponents(panel);

        frame.setVisible(true);
    }

    private static void placeComponents(JPanel panel) {
        panel.setLayout(null);

        JLabel labelA = new JLabel("Digite o valor de a:");
        labelA.setBounds(10, 20, 150, 25);
        panel.add(labelA);

        JTextField textA = new JTextField(20);
        textA.setBounds(170, 20, 165, 25);
        panel.add(textA);

        JLabel labelB = new JLabel("Digite o valor de b:");
        labelB.setBounds(10, 50, 150, 25);
        panel.add(labelB);

        JTextField textB = new JTextField(20);
        textB.setBounds(170, 50, 165, 25);
        panel.add(textB);

        JLabel labelN = new JLabel("Digite o valor de n:");
        labelN.setBounds(10, 80, 150, 25);
        panel.add(labelN);

        JTextField textN = new JTextField(20);
        textN.setBounds(170, 80, 165, 25);
        panel.add(textN);

        JLabel labelTol = new JLabel("Digite a tolerância:");
        labelTol.setBounds(10, 110, 150, 25);
        panel.add(labelTol);

        JTextField textTol = new JTextField(20);
        textTol.setBounds(170, 110, 165, 25);
        panel.add(textTol);

        JButton calculateButton = new JButton("Calcular");
        calculateButton.setBounds(10, 140, 150, 25);
        panel.add(calculateButton);

        JLabel resultLabel = new JLabel("Resultado:");
        resultLabel.setBounds(10, 170, 350, 25);
        panel.add(resultLabel);

        JProgressBar progressBar = new JProgressBar();
        progressBar.setBounds(10, 200, 325, 25);
        progressBar.setVisible(false);
        panel.add(progressBar);

        calculateButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                progressBar.setVisible(true);
                progressBar.setIndeterminate(true);

                SwingUtilities.invokeLater(() -> {
                    try {
                        BigDecimal a = new BigDecimal(textA.getText());
                        BigDecimal b = new BigDecimal(textB.getText());
                        int n = Integer.parseInt(textN.getText());
                        BigDecimal tol = new BigDecimal(textTol.getText());

                        MathContext mc = new MathContext(30, RoundingMode.HALF_UP);
                        BigDecimal result = trapezoidalRule(x -> BigDecimal.valueOf(Math.log10(Math.sqrt(x.doubleValue() * x.doubleValue() + x.doubleValue()) + 1)), a, b, n, tol, mc);
                        resultLabel.setText("Jadna meu amor, o valor que você calculou é: " + result.toPlainString());
                    } catch (Exception ex) {
                        resultLabel.setText("Ocorreu um erro: " + ex.getMessage());
                    } finally {
                        progressBar.setVisible(false);
                    }
                });
            }
        });
    }

    public static BigDecimal trapezoidalRule(Function<BigDecimal, BigDecimal> f, BigDecimal a, BigDecimal b, int n, BigDecimal tol, MathContext mc) {
        BigDecimal previousResult = BigDecimal.ZERO;
        BigDecimal currentResult = calculateTrapezoidal(f, a, b, n, mc);

        while (currentResult.subtract(previousResult).abs().compareTo(tol) > 0) {
            n *= 2; // Dobra o número de subintervalos para melhorar a precisão
            previousResult = currentResult;
            currentResult = calculateTrapezoidal(f, a, b, n, mc);
        }

        return currentResult;
    }

    public static BigDecimal calculateTrapezoidal(Function<BigDecimal, BigDecimal> f, BigDecimal a, BigDecimal b, int n, MathContext mc) {
        BigDecimal h = b.subtract(a).divide(BigDecimal.valueOf(n), mc);

        BigDecimal sum = IntStream.range(1, n)
                                  .parallel()
                                  .mapToObj(i -> f.apply(a.add(BigDecimal.valueOf(i).multiply(h, mc))))
                                  .reduce(BigDecimal.ZERO, BigDecimal::add);

        sum = sum.add(f.apply(a).add(f.apply(b)).divide(BigDecimal.valueOf(2), mc));
        return sum.multiply(h, mc);
    }
}
