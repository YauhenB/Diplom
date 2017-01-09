package by.grsu.diplom;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.presets.opencv_core;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.NetSaverLoaderUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

/**
 * Created by yauhen on 30.12.16.
 */
public class Trainer {
    private static final Logger log = LoggerFactory.getLogger(Trainer.class);

    public void trainNet(String path, long seed, MultiLayerNetwork network, int height, int width, int channels,
                         int batchSize, int numLabels, int epochs, int nCores, boolean save, String networkName) throws Exception {
        Random rng = new Random(seed);
        //loading files
        log.info("Load data....");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(path);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        //generating train data
        InputSplit[] inputSplit = fileSplit.sample(pathFilter);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[0];
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        log.info("Build model....");


        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;

        //creating model
        network.init();
        network.setListeners(new ScoreIterationListener(1));

        //Stats
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage));

        log.info("Train model....");


        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
        network.fit(trainIter);


        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));


        if (save) {


            log.info("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
//            NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath);
//            NetSaverLoaderUtils.saveUpdators(network, basePath);
            File locationToSave = new File(networkName);
            boolean saveUpdater = true;
            ModelSerializer.writeModel(network, locationToSave, saveUpdater);

            log.info("Saved");

        }

    }


}
