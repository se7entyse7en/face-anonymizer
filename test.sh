if [[ "$1" == "static_checks" ]]; then
    echo "Running static checks."
    flake8 --statistics "`pwd`"
    isort -rc -c -q
elif [[ "$1" == "py" ]]; then
    echo "Running unittests."
    python -m unittest discover -c
fi
