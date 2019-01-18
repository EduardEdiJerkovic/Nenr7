package hr.fer.model;

public class Sample {
    private double[] point;
    private double[] classification;

    Sample(double[] point, double[] classification) {
        this.point = point;
        this.classification = classification;
    }

    public double[] getPoint() {
        return point;
    }

    public double[] getClassification() {
        return classification;
    }

    public String toString() {
        var result = "";
        for (var p : point) {
            result += p + "\t";
        }
        for (var c : classification) {
            result += (int) c + "\t";
        }
        return result;
    }

    public int[] getIntClassification() {
        int[] result = new int[classification.length];
        for (int i = 0; i < classification.length; ++i) {
            result[i] = (int) classification[i];
        }
        return result;
    }
}
