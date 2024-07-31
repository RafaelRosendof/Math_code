import turtle
from canvas import canvasSv

# Configurar a tela
screen = turtle.Screen()
screen.bgcolor("white")

# Criar a tartaruga para desenhar
wr = turtle.Turtle()
wr.fillcolor('red')
wr.speed(2)

# Desenhar o coração
wr.begin_fill()
wr.left(140)
wr.forward(113)

for i in range(200):
    wr.right(1)
    wr.forward(1)
wr.left(120)

for i in range(200):
    wr.right(1)
    wr.forward(1)

wr.forward(112)
wr.end_fill()

# Esconder a tartaruga
wr.ht()

# Configurar a tartaruga para escrever
wr.penup()
wr.goto(-70, -50)  # Mover a tartaruga para a posição para escrever o texto
wr.color("black")
wr.write("Jadna", font=("Arial", 24, "bold"))

# Manter a janela aberta
turtle.done()