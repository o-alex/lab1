name := "lab1"

organization := "se.kth.spark"

version := "1.0"

scalaVersion := "2.11.0"

//resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.1" % "provided"

mainClass in assembly := Some("se.kth.spark.lab1.task6.Main")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
