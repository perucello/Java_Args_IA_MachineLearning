package java_argumento_generative_ai;

import java.util.Scanner;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.DenseInstance;

public class SentimentAnalysis {

    public static void main(String[] args) throws Exception {
        // Carregar o conjunto de dados (arquivo .arff com textos rotulados)
        System.out.println("Carregando o conjunto de dados...");
        DataSource source = new DataSource("C:\\Users\\HOME\\Downloads\\Java_GenerativeAI\\Java_argumento_generative_ai\\src\\java_argumento_generative_ai\\sentiment_data_arguments.arff");
        Instances dataset = source.getDataSet();
        System.out.println("Conjunto de dados carregado com sucesso!");

        // Definir qual atributo será classificado (último no conjunto de dados)
        System.out.println("Definindo o atributo que será classificado...");
        dataset.setClassIndex(dataset.numAttributes() - 1);
        System.out.println("Atributo de classe definido: " + dataset.classAttribute().name());

        // Transformar atributos de texto em vetores de palavras
        System.out.println("Convertendo atributos de texto em vetores de palavras...");
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataset); // Configurar o filtro para o formato dos dados

        // Aplicar o filtro ao conjunto de dados
        Instances filteredData = Filter.useFilter(dataset, filter);
        System.out.println("Atributos de texto convertidos com sucesso!");

        // Criar e treinar o classificador NaiveBayes
        System.out.println("Treinando o modelo NaiveBayes...");
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(filteredData);
        System.out.println("Modelo treinado com sucesso!");

        // Ler nova frase do console
        Scanner scanner = new Scanner(System.in);
        System.out.println("Digite uma frase para classificar:");
        String newText = scanner.nextLine();
        scanner.close();

        System.out.println("Classificando o novo exemplo de texto: " + newText);

        // Criar uma instância para o novo texto
        Instances emptyDataset = new Instances(dataset, 0); // Criar um dataset vazio com a mesma estrutura
        emptyDataset.setClassIndex(dataset.classIndex()); // Definir o atributo de classe

        // Criar uma nova instância com a mesma estrutura
        Instance newInstance = new DenseInstance(emptyDataset.numAttributes());
        newInstance.setDataset(emptyDataset); // Definir o dataset da nova instância

        // Definir o valor do atributo de texto
        newInstance.setValue(emptyDataset.attribute(0), newText); // Atribuir o texto ao primeiro atributo (ajustar conforme necessário)

        // Adicionar a nova instância ao conjunto vazio
        emptyDataset.add(newInstance);

        // Aplicar o filtro ao novo conjunto de dados com a nova instância
        Instances filteredTestData = Filter.useFilter(emptyDataset, filter);

        // Verificar se o filteredTestData contém pelo menos uma instância
        if (filteredTestData.numInstances() > 0) {
            Instance filteredInstance = filteredTestData.instance(0); // Obter a instância filtrada

            // Classificar a nova instância
            double label = classifier.classifyInstance(filteredInstance);

            // Exibir resultado da classificação
            System.out.println("Classificação prevista: " + dataset.classAttribute().value((int) label));
        } else {
            System.out.println("O conjunto de dados filtrados está vazio.");
        }
    }
}
