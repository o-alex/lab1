package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sparkContext.textFile(filePath).toDF("col")

    //------------- Transformations
    val regexTokenizer = new RegexTokenizer().setInputCol("col").setOutputCol("tokens").setPattern(",")
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("allFeatures")
    val lSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("yearv").setIndices(Array(0))
    val myudf: Vector => Double = _.apply(0)
    val v2d = new Vector2DoubleUDF(myudf).setInputCol("yearv").setOutputCol("year2d")
    val minYear:Double = 1922
    val mylabler : Double => Double = {_ - minYear}
    val lShifter = new DoubleUDF(mylabler).setInputCol("year2d").setOutputCol("label")
    val fSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("features").setIndices(Array(0,1,2))
    val myLR = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.1);

    val lrStages = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val pipeline = new Pipeline().setStages(lrStages)

//    val paramGrid = new ParamGridBuilder().addGrid(myLR.maxIter, Array(3,5,8, 10,15, 20))
//      .addGrid(myLR.regParam, Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)).build()
    val paramGrid = new ParamGridBuilder().addGrid(myLR.maxIter, Array(3,5))
      .addGrid(myLR.regParam, Array(0.1, 0.2)).build()

    val cv = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val pipelineModel:CrossValidatorModel  = cv.fit(rawDF)
    val bestModel = pipelineModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel]
    println("regParam: "+bestModel.getRegParam);
    println("maxItr: "+bestModel.getMaxIter);
    println("RMSE: "+ bestModel.summary.rootMeanSquaredError)



    val testFilePath = "src/main/resources/test-data.txt"
    val testDF = sparkContext.textFile(testFilePath).toDF("col")
    var res =  pipelineModel.transform(testDF)
    res.show(5)
  }
}