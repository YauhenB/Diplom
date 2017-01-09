package by.grsu.diplom;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by yauhen on 30.12.16.
 */
public class Tester {
    protected static final String chr = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    protected static final HashMap<Integer, Character> outMap = new HashMap<>();

    //    protected static final Logger log = LoggerFactory.getLogger(Tester.class);
    static { //populate map
        for (int index = 0; index < 62; index++) {
            outMap.put(index, chr.charAt(index));
        }
    }

    public static void testNet(String path, MultiLayerNetwork network, long seed, int height, int width,
                               int channels, int batchSize, int numLabels) throws IOException {

        File mainPath = new File(path);
        Random rng = new Random(seed);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter);
        InputSplit testData = inputSplit[0];
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        try {
            recordReader.initialize(testData);
        } catch (IOException e) {
            e.printStackTrace();
        }
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        System.out.println(eval.stats());


    }

    public static void predict(String filePath, MultiLayerNetwork network, int height, int width, int channels) throws IOException {
        File file = new File(filePath);
        ImageLoader loader = new ImageLoader(height, width, channels);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        INDArray array = loader.asRowVector(file);
        scaler.transform(array);
        Integer predictedVal;

        predictedVal = network.predict(array)[0];

        System.out.println(outMap.get(predictedVal));
//        log.info(network.output(array).toString());

    }
}
