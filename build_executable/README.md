Automatically Building Executable Files from python programs for RaspberryPi with pyinstaller


In order to build executable files in an automated manner some form of a Continuous integration/Continuous Development (CICD) tool would provide immeasurable support. One such tool was added to github not that long ago and it is Github actions.

GitHub Actions is a continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline. You can create workflows that build and test every pull request to your repository, or deploy merged pull requests to production.

GitHub Actions goes beyond just DevOps and lets you run workflows when other events happen in your repository. For example, you can run a workflow to automatically add the appropriate labels whenever someone creates a new issue in your repository.

GitHub provides Linux, Windows, and macOS virtual machines to run your workflows, or you can host your own self-hosted runners in your own data center or cloud infrastructure.

Using this repository as a guide to build executable files is pretty simple. Navigate to .github/workflows/ (your workflows need to be uploaded to this folder path in your repository). In this folder there are several different .yml files. This section will focus on the executable.yml file.

The executable.yml file will create the desired raspberry pi image defined in line 11, install the packages you define in the commands starting at line 34 and then compress and upload the image to github. You can then download that image and using raspi imager to install the image with all files from the repository including the built executable.

The lines the user may need to edit in image.yml are 10-13 and 34-46

Once you have copied the file and its contents simply edit them as needed for your project.

```yaml
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  workflow_dispatch:
```

  This defines what events you want the regression to run on. Currently I have commented out having it run on push requests and pull requests and it only will run on demand (workflow_dispatch).
  
``` yaml 
env:
  file_to_build: algo_VTCN.py #file you want to build
  pi_image: raspios_lite_arm64:latest #image you want to use need 64 bit image for pytorch
  path_to_file: IIoT_Device_Guide_UGA_2023/exampleCode #folder that $file_to_build is in
  built_file_name: algo_VTCN #name of the built file
```

These are the environment variables for the file. I chose to use environment variables here so it's obvious what some of the options you may want to change are.

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repo #checkout code so most up-to-date version is accessible to the runner
      uses: actions/checkout@main
      with: 
        lfs: true #lfs is github large file storage system
    - name: checkout lfs object
      run: git lfs checkout
    - name: Build test
      uses: pguyot/arm-runner-action@main #this action emulates an ARM CPU and a desired operating system $pi_image
      id: Build-Python-exe
      with:
        image_additional_mb: 3584
        copy_repository_path: /home/pi
        copy_artifact_path: ${{env.path_to_file}}/dist/${{env.built_file_name}}
        base_image: ${{env.pi_image}}
        commands: |
            sudo apt-get update -y;
            pwd;
            apt-get install -y python3 python3-venv python3-dev python3-pip; 
            cd ${{env.path_to_file}};
            pip3 install torch;
            pip3 install --upgrade pip;
            pip3 install numpy --upgrade;
            pip3 install -U scikit-learn;
            pip3 install -U matplotlib;
            pip3 install pyinstaller;
            pip3 install nvidia-ml-py3
            pyinstaller --clean --onefile ${{env.file_to_build}};
    - uses: actions/upload-artifact@main
      with:
        name: exe_file
        path: ${{env.built_file_name}}
        if-no-files-found: error
        retention-days: 3
```

This is jobs section and it defines what you want the runner to do when the events above happen. you can think of all of these commands as if you are running the commands on a raspberry pi - that is because you are essentially doing just that. The pguyot/arm-runner-action@main is using QEMU to emulate an ARM CPU and the desired raspios and is able to create a working program with pyinstaller because it emulates both the CPU and OS.


 If you have edited the environment variables at the top of the file as suggested the only thing you need to edit here are the packages that need to be installed to build the program. Please note that pyinstaller is required to be installed for the program to be built. This job is specifically building a raspberry pi image with python3 installed as well as all of the packages installed. The upload-artifact section is grabbing the specific file we wanted built and uploading it to github.

Finding the packages compatible with raspberry pi is likely what will cause the most problems with building the program. For instance the default Pi OS is 32 bit and pytorch will not run on a 32 bit operating system and the error codes given does not make it immediately obvious what the error is so some trial and error is required. Some tips: stack overflow is your friend adding the flags to some of these installs was done at the suggestion of posts I found on stack overflow and without these flags they would not install properly. Also, ensure the package is compatible with the desired operating system.

```yaml

  upload:
    runs-on: raspi
    needs: build
    steps: 
      - name: Checkout repo
        uses: actions/checkout@main
      - uses: actions/download-artifact@main
        with:
          name: exe_file
      - run: cp ${{env.built_file_name}} /home/pi/
      - run: cd /home/pi
      - run: chmod +x ${{env.built_file_name}}
```

You can add this job if your device is connected to the internet after you install a self-hosted runner to the device. To add a self hosted runner your repository should be made private then to go settings>actions>runners and then 'add self hosted runner' and follow the steps provided on the page. Be sure to add a tag that is meaningful to the machine such as I did 'raspi'. This tells github which machine to run the job on if you have more than one self-hosted runners.



```yaml
     - name: Compress the release image
       run: |
         sudo xz -T 0 -v ${{ steps.Build-Python-exe.outputs.image }}
     - name: Upload the image artifact
       uses: actions/upload-artifact@v3
       with:
         name: build_image
         path: ${{ steps.Build-Python-exe.outputs.image }}.xz
         if-no-files-found: error
         retention-days: 10
```

These lines are included in the image.yml and are included to upload the build image to github once the job has been completed. This can be very useful if you have many devices you wish to run the program or that all require python and specific packages added to them. You can just download the .img.xz file - unzip the file and use raspi imager to image your SD cards and have all the files and repository in the starting image of the raspberry pi.