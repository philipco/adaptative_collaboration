for file in cifar10/*.pdf; do
    pdftoppm -png -singlefile "$file" "${file%.pdf}"
done

