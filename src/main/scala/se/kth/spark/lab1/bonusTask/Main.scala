package se.kth.spark.lab1.bonusTask

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}
import org.apache.spark.ml.linalg.Vector

object bonusTask {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("salman-bonus-task").setMaster("local")
//    val conf = new SparkConf().setAppName("salman-bonus-task")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

//    val testFilePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/a.txt"
//    val filePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/all.txt"

    val testFilePath = "src/main/resources/a.txt"
    val filePath = "src/main/resources/b.txt"
    val rawDF = sparkContext.textFile(filePath).toDF("col")
    rawDF.take(1)


    //------------- Transformations
    val regexTokenizer = new RegexTokenizer().setInputCol("col").setOutputCol("tokens").setPattern("\\W")

    val r = regexTokenizer.transform(rawDF)
    r.show(5)
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("allFeatures")
    val r2 = arr2Vect.transform(r)
    r2.show(2)

    val lSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("yearv").setIndices(Array(0))
    val r3 = lSlicer.transform(r2)
    r3.show(3)

    val myudf: Vector => Double = _.apply(0)
    val v2d = new Vector2DoubleUDF(myudf).setInputCol("yearv").setOutputCol("year2d")
    val r4 = v2d.transform(r3)
    r4.show(5)

    //scaling the year
    val minYear:Double = 1900
    val mylabler : Double => Double = {_ - minYear}
    val lShifter = new DoubleUDF(mylabler).setInputCol("year2d").setOutputCol("label")
    val r5 = lShifter.transform(r4)
    r5.show(5)

    val fSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("features").setIndices(Array(0,1,2))
    val r6 = fSlicer.transform(r5)
    r6.show(5)

    val myLR = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.1);

    var lrStages = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val pipeline = new Pipeline().setStages(lrStages)
    val pipelineModel:PipelineModel  = pipeline.fit(rawDF)
    System.exit(0)
    val lrModel = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel]
    println("RSME of the model is "+lrModel.summary.rootMeanSquaredError)


    val testDF = sparkContext.textFile(testFilePath).toDF("col")
    var res =  pipelineModel.transform(testDF)
    res.show(50)
    sc.stop()
  }
}