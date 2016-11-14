package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SQLContext}

import scala.collection.mutable
object Main {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sparkContext.textFile(filePath).toDF("col")
    rawDF.show(1)

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("col")
      .setOutputCol("tokens")
      .setPattern(",")

    println("\nTokenized values are ")
    //Step2: transform with tokenizer and show 5 rows
    var colsData = regexTokenizer.transform(rawDF)
    colsData.select("tokens").take(5).foreach(println)

    println("\nVectors are ")
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("allFeatures")
    var vectors = arr2Vect.transform(colsData)
    vectors.select("allFeatures").take(5).foreach(println)

    //Step4: extract the label(year) into a new column
    import org.apache.spark.sql.functions._
    val lSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("yearv")
    lSlicer.setIndices(Array(0))
    val data = lSlicer.transform(vectors)
    data.select("yearv").take(5).foreach(println)


    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val myudf: Vector => Double = _.apply(0)
    val v2d = new Vector2DoubleUDF(myudf)
    v2d.setInputCol("yearv").setOutputCol("year2d")
    val data2 = v2d.transform(data)
    data2.select("allFeatures", "year2d").show(5)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYear:Double = 1922
    val mylabler : Double => Double = {_ - minYear}
    val lShifter = new DoubleUDF(mylabler)
    lShifter.setInputCol("year2d").setOutputCol("label")
    val data3 = lShifter.transform(data2)
    data3.select("allFeatures", "label").show(5)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("features")
    fSlicer.setIndices(Array(0,1,2))
    val data4 = fSlicer.transform(data3)
    data4.select("label", "features").show(5)


    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)


    //Step10: transform data with the model
    val testFilePath = "src/main/resources/test-data.txt"
    val testDF = sparkContext.textFile(testFilePath).toDF("col")
    var res = pipelineModel.transform(testDF)
    res.show(5)

    //      .select("label", "col", "probability", "prediction")
    //      .collect()
    //      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
    //        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    //      }val filePath = "src/main/resources/millionsong.txt"


    //Step11: drop all columns from the dataframe other than label and features
    testDF.drop("col", "tokenks","allFeatures", "yearv", "year2d")
  }

}