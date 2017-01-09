package by.grsu.diplom;

import org.slf4j.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.LoggerFactory;


import java.util.Random;


public class Runner {
//    protected static final Logger log = LoggerFactory.getLogger(Runner.class);

    protected static int height = 50; //100
    protected static int width = 50; //100
    protected static int channels = 1;
    protected static int numLabels = 62;
    protected static int batchSize = 20;
    protected static long seed = 60;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 20; //50
    protected static int nCores = 15;
    protected static boolean save = true;


    public static void main(String[] args) throws Exception {
        MultiLayerNetwork network = Builder.getCustomModel(seed, iterations, numLabels, height, width, channels);
        network = Builder.loadNetwork("./CHARNET", network);
        switch (args[0]) {
            case ("recog"): {
                Tester.predict(args[1], network, 50, 50, 1);
                break;
            }
            case ("eval"): {
                Tester.testNet(args[1], network, seed, height, width, channels, batchSize, numLabels);
                break;
            }
            default: {
                System.err.println("WRONG INPUT");
                break;
            }
        }

        //testing
//        Tester.testNet("/home/yauhen/Downloads/Img/test",network,seed,height,width,channels,batchSize,62);

//     Tester.predict("/home/yauhen/Downloads/Img/GoodImg/Bmp/L/img022-00061.png",network,50,50,1);

//        Trainer trainer=new Trainer();
//        trainer.trainNet("/home/yauhen/Downloads/Img/GoodImg/Bmp",seed,network,height,
//                width,channels,batchSize,numLabels,epochs,nCores,save,"CHARNET");
    }


}