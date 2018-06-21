import os

from flask import Flask, render_template, request, send_from_directory, url_for, jsonify, session, redirect
from werkzeug import secure_filename
from logging import Formatter, FileHandler
from natsort import natsorted

base_dir = os.path.abspath(os.path.dirname(__file__))
handler = FileHandler(os.path.join(base_dir, 'log.txt'), encoding='utf8')
handler.setFormatter(Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"))


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        ALLOWED_EXTENSIONS={'.png', '.jpeg', '.jpg'},
        UPLOAD_DIR=os.path.join(base_dir, 'upload/'),
        OUTPUT_DIR=os.path.join(base_dir, 'output/'),
        # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    app.logger.addHandler(handler)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return render_template('index.html', active_step='1')

    @app.route('/mark/<filename>')
    def mark(filename):
        if filename in session.get('file_info', []):
            return render_template('mark.html', filename=filename, active_step='3')

    @app.route('/run_segmentation/<filename>', methods=['POST'])
    def run_segmentation(filename):
        mask_coordinates = [[int(json_coord["x"]), int(json_coord["y"])] for json_coord in request.get_json()]
        input_sequence = natsorted(session.get('file_info', []))

        # Perform segmentation for given input sequence with user-selected mask
        
        output_images = []

        return jsonify(output_images)

    @app.route('/uploaded')
    def uploaded():
        file_info = session.get('file_info', [])
        return render_template('upload.html', file_info=natsorted(file_info), active_step='2')

    @app.route('/fetch_image/<filename>')
    def fetch_image(filename):
        if filename in session.get('file_info', []):
            return send_from_directory(app.config['UPLOAD_DIR'], filename)
        else:
            return redirect(url_for('/'))

    @app.route('/upload_ajax', methods=['POST'])
    def upload_ajax():
        files = request.files.getlist("files[]")
        file_info = []
        for uploaded_file in files:
            if uploaded_file and os.path.splitext(uploaded_file.filename)[1] in app.config['ALLOWED_EXTENSIONS']:
                filename = secure_filename(uploaded_file.filename)
                app.logger.info('FileName: ' + filename)

                save_path = os.path.join(app.config['UPLOAD_DIR'], filename)
                uploaded_file.save(save_path)
                file_info.append(filename)
        session['file_info'] = file_info
        return jsonify(file_info)

    return app
