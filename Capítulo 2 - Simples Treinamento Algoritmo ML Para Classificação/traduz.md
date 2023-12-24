Antes de discutirmos o perceptron e algoritmos relacionados com mais detalhes, vamos fazer um breve tour pelos primórdios do ML. Tentando entender como funciona o cérebro biológico para projetar uma IA.

Warrn McCulloch e Pitts publicaram o primeiro conceito de uma célula cerebral simplificada, o chamado neurônio, em 1943.
Os neurônios biológicos são células nervosas interconectadas no cérebro que estão envolvidas no processamento e transmissão de sinais químicos e elétricos.

McCulloch e Pitts descreveram essas células nervosas como uma porta lógica simples com saídas binárias; múltiplos sinais chegam aos dendritos, eles são então integrados ao corpo celular e, se os sinais acumulados excederem um determinado limite; é gerado um sinal de saída que será transmitido pelo axônio.

Apenas alguns anos depois, rosenblatt publicou o primeiro conceito da regra de aprendizagem perceptron baseada no modelo perceptron MCP. com sua regra do perceptron, Rosenblatt propôs um algoritmo que aprenderia automaticamente os coeficientes de peso ideais que seriam então multiplicados pelos recursos de entrada para tomar a decisão se um neurônio dispara ou não.

More fomally,we can put the idea behind artificial neuron into the context of a binary classification task with two classes : 0 and 1.
We can then define a decision function , o(z), that takes a linear combination of certain input values x, and a corresponding weight vector w, where  is the so called net input.
Now, if the net input of a particular example x, is greater than a defined threshold ,0, we predict class 1 and class 0 otherwise