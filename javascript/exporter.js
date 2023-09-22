document.addEventListener('DOMContentLoaded', function() {
    onUiLoaded(function () {
        console.log('hello-exporter')
        let container = gradioApp().getElementById('controlnet');
        let fieldsets = container.querySelectorAll('fieldset');
            fieldsets.forEach(function (fieldset) {
                let label = fieldset.firstChild.nextElementSibling;
                let textContent = label.textContent;
                if (textContent === 'Control Type') {
                    let radios = fieldset.querySelectorAll('input[type="radio"]');
                    console.log(radios)
                    console.log(label)
                    console.log('>>>>>>>>>>>>')
                }


                // if (value) {
                //     radios.forEach(function (radio) {
                //
                //     });
                // }
                // radios.forEach(function (radio) {
                //     radio.addEventListener('change', function () {
                //         //
                //     });
                // });
            });
    });
});