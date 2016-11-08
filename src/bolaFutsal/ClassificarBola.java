package bolaFutsal;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class ClassificarBola {
	// PARÂMETROS PARA A VALIDAÇÃO CRUZADA - Holdout
		public static int PARTICOES = 3;
		public static int ITERACOES = 1;


		public static void main(String[] args) throws Exception {

			FileReader leitor = new FileReader("bola_futsal.arff");
			Instances bola_futsal = new Instances(leitor);
			bola_futsal.setClassIndex(3);
			bola_futsal = bola_futsal.resample(new Random());

			for (int i = 0; i < ITERACOES; i++) {

				// Obtendo as partições de treino e de teste
				Instances irisTreino = bola_futsal.trainCV(PARTICOES, i);
				Instances irisTeste = bola_futsal.testCV(PARTICOES, i);

				// Definindo os classificadores
				IBk vizinho = new IBk();
				IBk knn = new IBk(2);

				// Treinando os classificadores
				vizinho.buildClassifier(irisTreino);
				knn.buildClassifier(irisTreino);

				// Anotando os resultados - Saída para um arquivo csv
				System.out.println("Real;Vizinho;kNN(2)");
				for (int j = 0; j < irisTeste.numInstances(); j++) {

					// Obtendo o exemplo a ser classificado
					Instance exemplo = irisTeste.instance(j);

					System.out.print(exemplo.value(3)); // classe real
					exemplo.setClassMissing(); // removendo a classe

					double vizinhoRes = vizinho.classifyInstance(exemplo);
					double knnRes = knn.classifyInstance(exemplo);

					// respostas dos classificadores avaliados
					System.out.println(";" + vizinhoRes + ";" + knnRes);
		
				}
			}
		}
}
