# Installation of Atari for Gym

## WICHTIG

Im folgenden Video zeige ich die Installation des Python Pakets Gym (Atari).

### Python Lib

pip install gym[atari]==0.17

## ROMs

Guide: <https://github.com/openai/atari-py>

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory.
