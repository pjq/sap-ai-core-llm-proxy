plugins {
    id("java")
    id("application")
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "me.pjq"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}

application {
    mainClass.set("me.pjq.Main")
}

// Configure Shadow JAR (Fat JAR)
tasks.shadowJar {
    archiveBaseName.set("sap-ai-core-client")
    archiveClassifier.set("")
    archiveVersion.set("")
    
    // Include all dependencies
    mergeServiceFiles()
    
    // Set main class for executable JAR
    manifest {
        attributes["Main-Class"] = "me.pjq.Main"
    }
}

// Also create a library JAR without Main-Class (for use as dependency)
tasks.register<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowLibJar") {
    archiveBaseName.set("sap-ai-core-client-lib")
    archiveClassifier.set("")
    archiveVersion.set("")
    
    from(sourceSets.main.get().output)
    configurations = listOf(project.configurations.runtimeClasspath.get())
    
    // Exclude the Main class for library usage
    exclude("me/pjq/Main.class")
    
    mergeServiceFiles()
    
    // No Main-Class attribute for library JAR
}