package hr.fer;

import hr.fer.model.GeneticAlgorithm;
import hr.fer.model.Samples;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        var path = "C:\\Users\\domin\\Documents\\Edi\\Fer\\DIPLOMSKI\\NENR\\DZ\\nenr7\\src\\hr\\fer\\SampleFile.txt";
        int numberOfClasses = 3;

        var samples = new Samples(path, numberOfClasses);
        var ga = new GeneticAlgorithm(100, new int[]{2, 8, 3}, samples, 3000000, Math.pow(10, -7));
        ga.setCrossoverRatio(1, 1, 1);
        ga.setMutationRatio(1, 1);
        var nnResult = ga.run();
        System.out.println("Result: " + nnResult.getFitness());
        int correctClassification = samples.prediction(nnResult);
        System.out.println("Correct classified: " + correctClassification + " out of " + samples.getSaplesSize());
    }
}
