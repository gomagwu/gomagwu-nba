from flask import Flask, render_template, request
import proj

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def pro():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    # HTML -> .py
    if request.method == 'POST':
        (wrank, erank, econf, wconf,
         efirst, wfirst, esec, wsec,
         fconf, eastch, westch, nbach) = proj.predict()
    return render_template('predict.html',
                           westrank=wrank.to_html(), eastrank=erank.to_html(),
                           eastconf=econf.to_html(), westconf=wconf.to_html(),
                           eastfirst=efirst, westfirst=wfirst,
                           eastsec=esec, westsec=wsec, conff=fconf, echamp=eastch, wchamp=westch, nbachamp=nbach)


@app.route("/create", methods=['POST'])
def submit():
    if request.method == 'POST':
        random = request.form["Abbreviation"]
        (mt, wrank, erank, econf, wconf,
         east1st, wfirst, esec, wsec,
         fconf, eastch, westch, nbach, nbam) = proj.random_team(random)
    return render_template('create.html', random=random, my_team=mt.to_html(),
                           westrank=wrank.to_html(), eastrank=erank.to_html(),
                           eastconf=econf.to_html(), westconf=wconf.to_html(),
                           eastfirst=east1st, westfirst=wfirst,
                           eastsec=esec, westsec=wsec, conff=fconf, echamp=eastch, wchamp=westch, nbachamp=nbach, fmat=nbam)


if __name__ == '__main__':
    app.run(debug=True)
