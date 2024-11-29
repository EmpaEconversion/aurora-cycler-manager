var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

dagfuncs.pipelineComparatorCustom = function (pip1, pip2) {
    const pip1compare = makePipelineComparable(pip1);
    const pip2compare = makePipelineComparable(pip2);
    if (pip1compare === null && pip2compare === null) {
        return 0;
    }
    if (pip1compare === null) {
        return -1;
    }
    if (pip2compare === null) {
        return 1;
    }
    if (pip1compare === pip2compare) {
        return 0;
    }
    return pip1compare < pip2compare ? -1 : 1;
}

// function to convert somestring-int-int to somestring-zeropaddedint-zeropaddedint
function makePipelineComparable(pipeline) {
    if (pipeline === undefined || pipeline === null) {
        return null;
    }

    // Split the pipeline string by '-'
    let parts = pipeline.split('-');

    // Iterate over the parts and pad numbers with zeros
    for (let i = 0; i < parts.length; i++) {
        if (!isNaN(parts[i])) {
            parts[i] = parts[i].toString().padStart(3, '0');
        }
    }

    // Join the parts back together with '-'
    pipeline = parts.join('-');

    // Now split by "_" and put the first part at the end
    parts = pipeline.split('_');
    pipeline = parts.slice(1).join('_') + '_' + parts[0];
    return pipeline;
}