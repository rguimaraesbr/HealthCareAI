
# Aplicação de  Modelos de Redes Neurais Convolucionais na classificação de exames histológicos de Cancer de Mama

#### Aluno: [Robson Guimarães](https://github.com/rguimaraesbr)
#### Orientador: [Leonardo Forero Mendoza ](https://github.com/leofome8).

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/rguimaraesbr/HealthCareAI/blob/main/C%C3%B3pia_de_Analise_de_Dados_pacientes_Cancer_com_4_modelos_e_otimizacao.ipynb). <!-- caso não aplicável, remover esta linha -->


- Trabalhos relacionados: <!-- caso não aplicável, remover estas linhas -->
    - [Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases ](https://pubmed.ncbi.nlm.nih.gov/27563488/).
    - [On Convolutional Neural Networks and Transfer Learning for Classifying Breast Cancer on Histopathological Images Using GPU](https://www.researchgate.net/publication/343891882_On_Convolutional_Neural_Networks_and_Transfer_Learning_for_Classifying_Breast_Cancer_on_Histopathological_Images_Using_GPU).
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
    - [Breast cancer histopathology image classification using AlexNet](https://ieeexplore.ieee.org/abstract/document/9036160).
    - [Computer Assisted Diagnosis of Breast Cancer Using Histopathology Images and Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/9036160).

---

### Resumo

 O câncer de mama é a forma mais comum de câncer em mulheres, sendo o Carcinoma Ductal Invasivo (CDI) o subtipo mais comum. O diagnóstico rápido, preciso e precoce do câncer podmelhora a probabilidade de sobrevivência. Portanto identificar e categorizar com precisão os subtipos de câncer de mama é uma tarefa clínica importante, e métodos automatizados podem ser usados para economizar tempo e reduzir erros.

Neste projeto classificaremos o Carcinoma Ductal Invasivo em  benigno e maligno a partir de imagens histopatológicas. Para isso iremos aplicar Redes Neurais Covolucionais. Iremos utilizar uma base de dados do Kaggle de imagens histopatologicas obtidas de (https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

Como passo inicial realizaremos a análise do Dataset, depois aplicaremos diversos modelos de redes convolucionais para análise de um fragmento do dataset. A partir da comparação entres o resultados da aplicação das redes,escolheremos uma rede para aplicar todo o conjunto de amostras, explicitando os resultados obtidos.



### 1. Introdução
No Brasil, segundo o Instituto Nacional de Câncer (INCA), o câncer de mama  é o tipo de câncer que mais acomete as mulheres no país (excluídos os tumores de pele não melanoma). Para 2019, estimou-se 59.700 casos novos, o que representa uma taxa de incidência de 51,29 casos por 100 mil mulheres. Com uma taxa de 13,68 óbitos/100 mil mulheres em 2015, a mortalidade por câncer de mama (ajustada pela população mundial) apresenta uma curva ascendente e representa a primeira causa de morte por câncer nas mulheres brasileiras. O Carcinoma Ductal Invasivo (CDI) é de longe o subtipo de câncer de mama mais comum, representando 80% dos casos.

Tumores são feixes de células que não deveriam se agrupar e crescem em pedaços sólidos. Os tumores podem ser benignos (não cancerígenos) e por ser confinados a uma região específica podem não causar problemas. Eles podem crescer e causar problemas devido ao tamanho. Se um tumor começa a crescer fora do grupo de células - torna-se maligno (cancerígeno). O câncer pode invadir o tecido local ou metastatizar e atacar outros tecidos. Temos uma gama de informação sobre tumores e câncer, incluindo subtipos e seus graus, mas o conjunto de dados com o qual estamos trabalhando simplesmente classifica as imagens como não cancerígenas (benignas) e cancerosas (malignas). 

Muitas vezes, o câncer é fisicamente perceptível no tecido e pode ser mais facilmente tratável quando detectado precocemente. A Histologia estuda os tecidos e a Patologia estuda as doenças. A histopatologia é a ciencia que estuda as doenças em tecidos,  patologistas examinam imagens de tecido (imagens de histologia) e chegam a um veredicto. De certa forma, os patologistas estão realizando a classificação (positiva ou negativa) com base em padrões e ocorrências nas imagens (características visuais). Este é um processo longo e trabalhoso, requerendo muita experiência.

![JPI-7-29-g001](https://user-images.githubusercontent.com/79609143/209991777-f05452e8-5714-4626-a9fb-c87ce9e2f2fd.jpg)

Algumas áreas podem não ter equipamentos ou recursos humanos necessários para se ter diagnóstico de forma rápida, fazendo com que os pacientes tenham que se deslocar para serem diagnosticados, prolongando o período em que não podem receber tratamento. Entretanto o carcinoma ductal invasivo (IDC) é bastante curável sendo que a taxa de sobrevivência de cinco anos é quase 100% quando tratado precocemente. Se o câncer se espalha para outros tecidos da região, esta taxa é de 86% e caso tenha se  espalhado para áreas distantes do  corpo,  é de 28%.

A aplicação do aprendizado de máquina na medicina é vasta e um tópico extremamente complexo por si só,  algumas das principais áreas incluem:

    Medicina de Precisão - Adaptação de medicamentos para indivíduos
    Diagnóstico por imagem médica -Diagnóstico de doenças com base em imagens, etc.)
    Descoberta de drogas - geração de estruturas como proteínas ou moléculas semelhantes a drogas, previsão de bioatividade, etc.

Descobrir se alguém sofre de uma determinada doença é difícil. Leva anos de prática, intuição e experiência para diagnosticar com um relativo nível de certeza se alguém sofre de uma condição ou não com base em imagens médicas. Automatizar esse processo tem implicações significativas para a velocidade do diagnóstico - e quanto mais rápido alguém for diagnosticado, mais rápido poderá receber tratamento. Em alguns casos, esse tempo pode ser essencial.


### 2. Modelagem

#### 2.1 Dataset
O Dataset que vamos trabalhar vem de um estudo  2016 "Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases" by Andrew Janowczyk and Anant Madabhushi. Dentre os casos de uso,  temos a classificação de IDC, para a qual eles tiveram uma pontuação F de 0,7648 em 50k patches de teste. O dataset refere-se a 279 pacientes, cada um com um ID exclusivo. 

Cada paciente tem uma pasta dedicada, nomeada por seu ID, com duas subpastas - 0 e 1. A pasta chamada 0 consiste em imagens de amostras de tecido benigno (aquelas sem marcadores IDC). A pasta denominada 1 consiste em imagens de amostras de tecidos malignos (aquelas que contêm marcadores IDC).

Cada patch tem um formato de nome distinto - uxXyYclassC.png, onde u é o ID do paciente, x é a coordenada X da qual o patch foi extraído, y é a coordenada Y da qual o patch foi extraído e a classe é 0 ou 1, denotando se os marcadores IDC estão presentes ou não naquele patch.

Fazendo a reconstrução de uma imagem so com base nos arquivos, conseguimos ver a composição

![breast cancer](https://user-images.githubusercontent.com/79609143/210016132-a2059f90-abd2-4350-8b02-81b9abd0687c.png)

Após analisar todo o dataset obtivemos a seguinte distribuição

![class imbalance](https://user-images.githubusercontent.com/79609143/210016827-4140508d-4fea-442a-90e6-99478ae6da21.png)

Verficamos um desbalanceamento entre os dados, sendo que a classe negativa equivale a 71% dos dados. Isto decorre principalmente devido a incidência de cancer na população e também porque os cortes de um tumor correspondem a partes de uma imagem histólogica completa.Será mais difícil orientar as redes para classificarem com precisão as instâncias IDC(+) em suas representações de conhecimento interno, pois há um incentivo maior em enviar previsões negativas para a maioria das entradas.

Para este trabalho não iremos realizar tentativas de balanceamento do dataset como undersampling ou oversampling, visto que poderia comprometer a qualidade dos dados para treinamento e como não temos acesso a um patologista, não teriamos como contrapor a nova base com alguma visão tecnica. Entretanto para minimizar o desbalanceamento dos dados no treinamento aumentamos os pesos das amostras positivas em relação as positivas.

#### 2.2 Aplicação do Modelo 
Existe uma carência de redes e  bases de imagens histológicas, com isso utilizamos as redes ja pré treinadas com Transfer Learning 

Podemos citar como vantagens dessa abordagem:
- • Pode-se usar modelos que foram cuidadosamente projetados por especialistas;
- • Como os especialistas criaram esses modelos, não é necessario se preocupar com qual arquitetura ou camadas usar ou incluir;
- • Devido ao seu design cuidadoso, eles tendem a ter um bom desempenho em detecção de imagem.

Nesse trabalho consideramos 4 redes: uma rede escrita "from scratch", EfficientNet0, Resnet e Xception da Google

A partir de uma análise comparativa da utilização destas redes como classificadores numa amostra reduzida de todo o dataset, escolhemos uma rede para rodar o dataset integralmente de forma otimizada.

Para fins de comparação da aplicabilidade das redes utilizamos uma amostra menor do dataset.  Segue o resultado comparativo 

![comparacao](https://user-images.githubusercontent.com/79609143/214692337-0f17d4d9-a2c6-4e9f-bac6-65528e55fbc5.png)


Escolhemos para seguir adiante o Effnet, devido a ter o melhor recall e AUC equivalente ao Xception, além exigir menor capacidade de computação.

### 3. Resultados

Para a avaliação dos resultados é necessário definir o significado de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos. Verdadeiro positivos e verdadeiros negativos são classificações corretas, ou seja, um tumor maligno ou um tumor benigno classificado corretamente. Em contraste, falso positivo e falso negativo são classificações erradas, ou seja, tumores classificados inversamente (um tumor maligno que deve ser benigno e vice-versa).  Podemos então definir as métricas de avaliação:
#### Precisão 
Essa métrica é a razão entre verdadeiros positivos e verdadeiros positivos além de falsos positivos. Assim, uma baixa precisão indica que o número de classificações corretas (malignos) é muito baixo, ou o número de falsos positivos (tumores benignos classificados como malignos) é alta

![precisao](https://user-images.githubusercontent.com/79609143/209995973-dbad9549-2765-4a4f-8dbf-290a841de542.png)

#### Recall
É a razão entre os verdadeiros positivos e verdadeiros positivos mais falsos positivos. Esta metrica indica que o algoritmo está funcionando bem na identificação de verdadeiros positivos. Por outro lado, se esta métrica for baixa,pode significar que um grande número de tumores benignos está sendo classificado de forma errada. Assim, esta métrica é fundamental para minimizar onúmero de falsos negativos, que é o pior cenário para o paciente.

![recall](https://user-images.githubusercontent.com/79609143/210002729-11060673-d8ce-47be-b93c-bbf5202ccc5f.png)

### F1-Score
É a media hamônica entre precisão e recall. Se olharmos para précisão e recall, essas métricas visam ajudar a detectar tumores malignos.No entanto, a precisão é sensível a falsos positivos, enquanto o recall é afetada por falsos negativos. Assim, o F1 Score fornece uma visão geral, em que valores altos podem indicar que o modelo tem um bom desempenho, não dando falsas classificações

![f1score](https://user-images.githubusercontent.com/79609143/210003249-3ba596de-3af3-45c9-8e08-40f027e59bfb.png)

Devido a decisão de não balancear as classes é provável que a precisão seja uma métrica ruim para o desempenho da rede, porque 71% de precisão seria otimo considerando o desequilíbrio entre as classes, portanto usaremos a metrica F1 Score além da precisão para ajustar o modelo

Como a F1 score é baseado na média harmônica entre recall e precisão e o Keras não possui uma métrica F1 Score integrada, definiremos nossa própria métrica F1 score para o modelo final.Idealmente encontraríamos uma função de perda para alinhar com F1, mas não existe tal função de perda incorporada, portanto a Binary Crossentropy foi usada, pois estamos realizando a classificação binária.


Conforme tabela abaixo, foi obtido apos a otimização um F1 Score de 89% para negativo e 71% para positivo, refletindo ainda o desbalanceamento da base. 

![resultado final](https://user-images.githubusercontent.com/79609143/214695251-8067439d-bc9e-4170-aa64-e54168938c33.png)




### 4. Conclusões

Este trabalho utilizou redes neurais convolucionais em um banco de dados de xxx imagens histopatologicas de xx pacientes, classificando-os em duas classes: cancer e nao cancer. Durante o treinamento das redes, utilizamos o transfer learning das seguintes arquiteturas de redes neurais convolucionais: from scratch", EfficientNet0, Resnet e Xception para uma pequena porçao dos dados.Os resultados mostraram que as diferentes estruturas convolucionais  abordam de maneira competente o problema de classificação através da transferência de aprendizado. Escolhemos a EfficientNet0 para poder rodar o total das amostras,  devido a que possui menor numero de parametros, apresentando custo beneficio adequado. 

Realizamos o tunning dos parametros da rede selecionada, utilizando a ferramenta de tunning do Keras. Foi verificado entretanto pouca melhoria nas metricas de avaliação do aprendizado da rede. Nossa concluão e que o maior ofensor para melhoria era o desbalanceamento da base de dados que apresentava um total de 20% de amostras positivas somente, o que era esperado devido a natureza do problema.

Nossos resultados mostram o grande potencial do uso de técnicas de deep learning combinadas com imagens histopatologicas no auxílio ao diagnóstico do câncer de mama. Como trabalho futuro indicamos a utilização de redes mais potentes como por exemplo EfficientNetB2 a EfficientNetB7  e estudos para melhoria no desabalanceamento da base de dados em questão, utilizando por exemplo a bibiloteca inbalance trabalhando em conjunto com um patologista.




---

Matrícula: 211.000.225

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
