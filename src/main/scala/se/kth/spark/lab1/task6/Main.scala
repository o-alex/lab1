package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PolynomialExpansion, RegexTokenizer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local[6]")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

//    def gradientSummand(weights: Vector, lp: Instance): Vector = {
//      val pred = Helper.predictOne(weights,lp.features)
//      val actual = lp.label
//      VectorHelper.dot(lp.features, (pred-actual))
//    }
//
//    val testdata = Array((Instance(1.0, Vectors.dense(1.0,1.0))), (Instance(2.0, Vectors.dense(2.0,2.0))))
//    val rdd = sc.parallelize(testdata)
//    val weights = Vectors.dense(Array(3.0,2.0))
//    val lp = new Instance(1.0, Vectors.dense(1.0,1.0))
//    val lp2 = new Instance(2.0, Vectors.dense(2.0,2.0))
//    val r = gradientSummand(weights, lp)
//    r.toArray.foreach(println)
//    val r2 = gradientSummand(weights, lp2)
//    r2.toArray.foreach(println)
//    def gradient(d: RDD[Instance], weights: Vector): Vector = {
//      d.map(x=>gradientSummand(weights,x)).reduce((x,y)=>VectorHelper.sum(x,y))
//    }
//
//    val r3 = gradient(rdd,weights)
//    r3.toArray.foreach(println)
//    System.exit(0)

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
    val fSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("features").setIndices(Array(1,2,3))
    val myLR = new MyLinearRegressionImpl()

    var lrStages = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val pipeline = new Pipeline().setStages(lrStages)
    val pipelineModel:PipelineModel  = pipeline.fit(rawDF)
    val lrModel = pipelineModel.stages(6).asInstanceOf[MyLinearModelImpl]
    println("Training error ")
    lrModel.trainingError.foreach(println)


    val testFilePath = "src/main/resources/test-data.txt"
    val testDF = sparkContext.textFile(testFilePath).toDF("col")
    var res =  pipelineModel.transform(testDF)
    res.show(5)

//    val filePath = "src/main/resources/millionsong.txt"
//    val obsDF: DataFrame = ???
//
//    val myLR = ???
//    val lrStage = ???
//    val pipelineModel: PipelineModel = ???
//    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    //print rmse of our model
    //do prediction - print first k
  }
}