<h1 align="center"> tinyjam </h1>
<h3 align="center"><img src="./assets/queen.png" width="700px"></h3>
<p align="center"> 💃🏽- jam 2 tiny desks </p>


This is a script to download/play tiny desks in the way I like them.

## Installation

### Brew
If you are running MacOS & have `brew` install & run like this
```
brew tap aryakaul/formulae
brew install tinyjam
wget "https://raw.githubusercontent.com/aryakaul/tinyjam/main/arya-curated"
tinyjam -l ./arya-curated -n
```

### git
Install `mpv` & `yt-dlp`. Then,
```
git clone "https://github.com/aryakaul/tinyjam.git"
cd tinyjam
wget "https://raw.githubusercontent.com/aryakaul/tinyjam/main/arya-curated"
./tinyjam -l ./arya-curated -n
```

## Usage

```
➜ tinyjam -h

    Jam to tiny desks with tinyjam v0.1.2!
        -l,--list       path to a file where every line is the name of an artist 
                        who appeared on tiny desk. if they did multiple shows
                        adding the year they performed can specify this.
        -o,--output     path to an output directory to download video files to,
                        default is './jamsesh'
        -n,--nodownload do not download video files, instead stream directly 
                        from youtube
        -f,--force      if output directory exists + contains same # of elements as
                        the number of lines in list then nuke contents and
                        download videos again. default behavior is to assume all
                        files are already downloaded and play them
        -c,--color      flag to display videos in color
        -h,--help       display this help message
```

