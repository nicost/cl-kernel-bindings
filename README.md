Bindings to Robert Haase's opencl kernels for image processing.

This code enables using these kernels without depending on ImageJ and all its brathers and sisters..

Development takes place in the master branch.  The oldClij branch continues 
working with the clij/clij-opencl-kernels code, wherease the master branch
uses kernels from https://github.com/nicost/clij-opencl-kernels.  

To use this code in your projects, add the following repository:

<repository>
	<snapshots>
		<enabled>true</enabled>
	</snapshots>
	<id>valenico</id>
	<url>https://valelab4.ucsf.edu/nstuurman/maven</url>
</repository>

and the following dependency:

<dependency>
	<groupId>org.micromanager</groupId>
	<artifactId>clij-opencl-kernels</artifactId>
	<version>0.1-SNAPSHOT</version>
</dependency>




