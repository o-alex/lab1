package se.kth.spark.lab1.task1

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Row, SQLContext}

object task1 {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
//    val rawDF = ???

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5).foreach(x=>{ println(x)} )

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(x=> x.split(","))
    recordsRdd.take(5).foreach(x=> println(x.deep.mkString("|")))

    case class Song(date:Double, f1:Double, f2:Double, f3:Double){
      def printInfo() : Unit = {
        println(s"year: $date f1: $f1 f2:$f2 f3:$f3")
      }
    }

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(x=>{Song(x(0).toDouble,x(1).toDouble,x(2).toDouble,x(3).toDouble)})
    songsRdd.take(5).foreach(x=>x.printInfo())


    //Step4: convert your rdd into a datafram
    //example code taken from https://spark.apache.org/docs/1.6.1/sql-programming-guide.html
    // Import Row.
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.types.{StructType,StructField,StringType};

    // Generate the schema based on the string of schema
    val schemaString = "date f1 f2 f3"
    val fields = schemaString.split(" ").map(field => StructField(field, DoubleType, nullable = true ))
    val schema = StructType(fields)
    val rowRDD = songsRdd.map(song => Row(song.date, song.f1, song.f2, song.f3))

    val songsDf = sqlContext.createDataFrame(rowRDD, schema)
    songsDf.createOrReplaceTempView("songs")
    println("Total Songs in the dataset are ")
    var result = sparkSession.sql("Select count(*) from songs ")
    result.show()


    println("Total song between 1998 and 2000")
    result = sparkSession.sql("Select count(*) from songs where date > 1998 and date < 2000 ")
    result.show()


    println("Year max value ")
    result = sparkSession.sql("Select max(date) from songs")
    result.show()
    println("Year min value ")
    result = sparkSession.sql("Select min(date) from songs")
    result.show()

    println("Songs between 2000 and 2010")
    result = sparkSession.sql("Select count(*) from songs where date > 2000 and date < 2010")
    result.show()
}
}
