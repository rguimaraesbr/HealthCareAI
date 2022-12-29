
# Aplicação de  Modelos de Redes Neurais Convolucionais na classificação de exames histológicos de Cancer de Mama

#### Aluno: [Robson Guimarães](https://github.com/rguimaraesbr)
#### Orientadora: [Leonardo ](https://github.com/link_do_github).

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/HealthCareAI). <!-- caso não aplicável, remover esta linha -->


- Trabalhos relacionados: <!-- caso não aplicável, remover estas linhas -->
    - [Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases ](https://pubmed.ncbi.nlm.nih.gov/27563488/).
    - [Nome do Trabalho 2](https://link_do_trabalho.com).
    

---

### Resumo

O Carcinoma Ductal Invasivo (CDI) é o subtipo mais comum de todos os cânceres de mama. O câncer de mama é a forma mais comum de câncer em mulheres. Identificar e categorizar com precisão os subtipos de câncer de mama é uma tarefa clínica importante, e métodos automatizados podem ser usados para economizar tempo e reduzir erros.O diagnóstico rápido, preciso e precoce do câncer melhora a probabilidade de sobrevivência. 

Neste projeto classificaremos o Carcinoma Ductal Invasivo em  benigno e maligno a partir de imagens histopatológicas. Para isso iremos aplicar Redes Neurais Covolucionais. Iremos utilizar uma base de dados do Kaggle de imagens histopatologicas utilizadas inicialmente em xx.

Como passo inicial realizaremos a análise do Dataset, depois aplicaremos diversos modelos de redes convolucionais para análise de um fragmento do dataset. A partir da comparação entres o resultados da aplicação das redes,escolheremos uma rede para aplicar todo o conjunto de amostras, explicitando os resultados obtidos.



### 1. Introdução
No Brasil, segundo o Instituto Nacional de Câncer (INCA), o câncer de mama  é o tipo de câncer que mais acomete as mulheres no país (excluídos os tumores de pele não melanoma). Para 2019, foram estimados 59.700 casos novos, o que representa uma taxa de incidência de 51,29 casos por 100 mil mulheres. Com uma taxa de 13,68 óbitos/100 mil mulheres em 2015, a mortalidade por câncer de mama (ajustada pela população mundial) apresenta uma curva ascendente e representa a primeira causa de morte por câncer nas mulheres brasileiras.

Tumores são feixes de células que não deveriam se agrupar e crescem em pedaços sólidos. Os tumores podem ser benignos (não cancerígenos) e confinados a uma região específica e podem não causar problemas. Eles podem crescer e causar problemas devido ao tamanho. Se um tumor começa a crescer fora do grupo de células - torna-se maligno (cancerígeno). O câncer pode invadir o tecido local ou metastatizar e atacar outros tecidos. Temos uma gama de informação sobre tumores e câncer, incluindo subtipos e seus graus, mas o conjunto de dados com o qual estamos trabalhando simplesmente classifica as imagens como não cancerígenas (benignas) e cancerosas (malignas). O Carcinoma Ductal Invasivo (CDI) é de longe o subtipo de câncer de mama mais comum, representando 80% dos casos.

Muitas vezes, o câncer é fisicamente perceptível no tecido e pode ser mais facilmente tratável quando detectado precocemente. A Histologia estuda os tecidos e a Patologia estuda as doenças. A histopatologia é a ciencia que estuda as doenças em tecidos,  patologistas examinam imagens de tecido (imagens de histologia) e chegam a um veredicto. De certa forma, os patologistas estão realizando a classificação (positiva ou negativa) com base em padrões e ocorrências nas imagens (características visuais). Este é um processo longo e trabalhoso, requerendo muita experiência.

![JPI-7-29-g001](https://user-images.githubusercontent.com/79609143/209991777-f05452e8-5714-4626-a9fb-c87ce9e2f2fd.jpg)

Algumas áreas podem não ter equipamentos ou recursos humanos necessários para se ter diagnóstico de forma rápida, fazendo com que os pacientes tenham que se deslocar para serem diagnosticados, prolongando o período em que não podem receber tratamento. Entretanto o carcinoma ductal invasivo (IDC) é bastante curável, principalmente quando detectado e tratado precocemente. A taxa de sobrevivência de cinco anos para o carcinoma ductal invasivo localizado é alta - quase 100% quando tratado precocemente. Se o câncer se espalha para outros tecidos da região, esta taxa é de 86% e caso tenha se  espalhado para áreas distantes do  corpo,  é de 28%.

## 1.2 Aplicação de inteligencia artificial na medicina

A Inteligencia Artifial tem sido cada vez mais empregada na medicina e está ajudando a salvar vidas de uma ampla variedade de condições médicas. 

A aplicação do aprendizado de máquina na medicina é vasta e um tópico extremamente complexo por si só,  algumas das principais áreas incluem:

    Medicina de Precisão - Adaptação de medicamentos para indivíduos
    Diagnóstico por imagem médica -Diagnóstico de doenças com base em imagens, etc.)
    Descoberta de drogas - geração de estruturas como proteínas ou moléculas semelhantes a drogas, previsão de bioatividade, etc.


Descobrir se alguém sofre de uma determinada doença é difícil. Leva anos de prática, intuição e experiência para diagnosticar com um relativo nível de certeza se alguém sofre de uma condição ou não com base em imagens médicas. Automatizar esse processo tem implicações significativas para a velocidade do diagnóstico - e quanto mais rápido alguém for diagnosticado, mais rápido poderá receber tratamento. Em alguns casos, esse tempo pode ser essencial.


## 1.3 Redes Convolucionais e Transfer Learning aplicadas ao diagnostico de cancer de mama

O Transfer Learning xxx   

Existe uma carencia de bases de imagens histologicas, com isso utilizaremos as redes ja pre treinadas e rodaremos em cima de uma amostra percentual de todo o dataset. 

Nesse trabalho consideramos as seguintes redes covolucionais 

1 - Basica, criada a partir x - descricao sucinta

2 - EfficientNet0 - descricao sucinta com referencia

3 - Resnet - descricao sucinta com referencia

4 - xxx - descricao sucinta com referencia




Final da Introducao



Proin feugiat nulla sem. Phasellus consequat tellus a ex aliquet, quis convallis turpis blandit. Quisque auctor condimentum justo vitae pulvinar. Donec in dictum purus. Vivamus vitae aliquam ligula, at suscipit ipsum. Quisque in dolor auctor tortor facilisis maximus. Donec dapibus leo sed tincidunt aliquam.

### 2. Modelagem

Modelagem dos dados

Resumo das redes Convolucionais

Comparacao entre os modelos 

Escolha da rede, porque?

### 3. Resultados

Aplicaçao da rede 

Otimizacao de parametros

Resultados


### 4. Conclusões

Este trabalho utilizou redes neurais convolucionais em um banco de dados de xxx imagens histopatologicas de xx pacientes, classificando-os em duas classes: cancer e nao cancer. Durante o treinamento das redes, utilizamos o transfer learning das seguintes arquiteturas de redes neurais convolucionais: AlexNet, GoogLeNet, ResNet-18, VGG-16 e VGG-19 para uma pequena porçao dos dados. Escolhemos a EfficientNet0 para poder rodar o total das amostras 

Nossos resultados mostram o grande potencial do uso de técnicas de deep learning combinadas com imagens histopatologicas no auxílio ao diagnóstico do câncer de mama.


Conclusao
Proximos trabalhos

---

Matrícula: 123.456.789

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
